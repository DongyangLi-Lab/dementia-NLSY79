"""Fold-fitted preprocessing and persistent endpoint/conditional-change models.

Adapted from BASE benchmark.py and the local fitted_models_review_v1 lifecycle.
See planning/CODE_ADAPTATION.md for the retained ideas and changed estimands.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import categorical_columns, require


class FittedRegressor:
    """Persist the feature schema, preprocessor and regressor as one object."""

    def __init__(self, learner, features, params, seed=42, threads=4):
        self.learner = learner
        self.features = list(features)
        self.params = dict(params)
        self.seed = seed
        self.threads = threads
        self.categories = categorical_columns()

    def frame(self, data):
        require(set(self.features) <= set(data), 'Required predictors are missing')
        x = data.loc[:, self.features].copy()
        groups = {}
        for column in self.features:
            if '__' in column:
                groups.setdefault(column.split('__')[0], []).append(column)
        # Preserve missing categorical parents, including when a target-only level appears.
        for parent, columns in groups.items():
            values = x[columns]
            observed = values.notna().all(axis=1)
            require((values.isna().all(axis=1) | observed).all(), 'Partly missing one-hot group: ' + parent)
            require(values.loc[observed].isin([0, 1]).all().all(), 'Invalid one-hot category')
            require(values.loc[observed].sum(axis=1).eq(1).all(), 'Incomplete one-hot group: ' + parent)
            codes = values.fillna(0).idxmax(axis=1).str.split('__').str[-1].where(observed, 'MISSING')
            x = x.drop(columns=columns)
            x[parent] = codes.astype(str)
        for column in self.categories:
            if column in x:
                x[column] = x[column].map(lambda value: 'MISSING' if pd.isna(value) else str(float(value)))
        if self.learner == 'trajectory':
            x = x.drop(columns='followup_years')
        return x

    def design(self, data, fit=False):
        x = self.frame(data)
        if fit:
            categories = list(x.select_dtypes(include=['object', 'string']).columns)
            numeric = [c for c in x if c not in categories]
            self.preprocessor = ColumnTransformer([
                ('numeric', Pipeline([
                    ('impute', SimpleImputer(strategy='median', keep_empty_features=True, add_indicator=True)),
                    ('scale', StandardScaler()),
                ]), numeric),
                ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categories),
            ])
            z = self.preprocessor.fit_transform(x)
        else:
            z = self.preprocessor.transform(x)
        if self.learner == 'trajectory':
            # A conditional mean change curve with delta(X, 0) exactly zero.
            t = data.followup_years.to_numpy(dtype=float).reshape(-1, 1) / 10.0
            z = np.column_stack([t, t*t, t*z])
        require(np.isfinite(z).all(), 'Nonfinite model design matrix')
        return z

    def fit(self, data, y, sample_weight=None):
        z = self.design(data, fit=True)
        y = np.asarray(y, dtype=float)
        if self.learner == 'elasticnet':
            self.model = ElasticNet(max_iter=50000, tol=1e-5, random_state=self.seed, **self.params)
        elif self.learner in ['ridge', 'trajectory']:
            self.model = Ridge(fit_intercept=self.learner != 'trajectory', **self.params)
        elif self.learner == 'catboost':
            from catboost import CatBoostRegressor
            self.model = CatBoostRegressor(loss_function='RMSE', random_seed=self.seed,
                                          thread_count=self.threads, verbose=False,
                                          allow_writing_files=False, **self.params)
        elif self.learner == 'lightgbm':
            from lightgbm import LGBMRegressor
            self.model = LGBMRegressor(objective='regression', random_state=self.seed,
                                      n_jobs=self.threads, verbosity=-1,
                                      deterministic=True, force_col_wise=True, **self.params)
        else:
            raise ValueError('Unknown learner: ' + self.learner)
        target = y - data.baseline_cognition27.to_numpy() if self.learner == 'trajectory' else y
        self.model.fit(z, target, sample_weight=sample_weight)
        return self

    def predict(self, data):
        prediction = np.asarray(self.model.predict(self.design(data)), dtype=float)
        if self.learner == 'trajectory':
            prediction += data.baseline_cognition27.to_numpy()
        require(np.isfinite(prediction).all(), 'Nonfinite prediction')
        return prediction
