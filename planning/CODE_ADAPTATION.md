# Adaptation from the HRS project

## Sources actually inspected

The source branch is `BASE` at the commit recorded in `code_provenance.json`. Its `benchmark.py` provides the model registry, training-only preprocessing, development-only hyperparameter selection, separate external prediction, and environment recording pattern.

The existing local `output/fitted_models_review_v1/train_and_save.py` and `model_store.py` already save fitted models and imputers, feature schemas, checksums, and prediction-replay checks. Those implementations and their existing six-year/eight-year artifacts were inspected. Their lifecycle is adapted into `part2_models/store.py`; unused neural networks, figures, notebooks, and old results are not copied into this branch.

## Changes for this task

| HRS paper workflow | Part 2 implementation |
| --- | --- |
| Six/eight-year dementia classification | Continuous 27-point cognition with actual elapsed years as an input |
| Classifier and `predict_proba` | ElasticNet, CatBoostRegressor, LGBMRegressor, and `predict` |
| Original paper's feature schema | Exact shared 29/77 and NLSY79 additional 82 candidate lists |
| Repeated individual random splits | Frozen household folds; HRS person-balanced pair weights |
| Validation AUPRC and probability thresholds | Development CV RMSE; no classification threshold |
| Development-to-temporal HRS validation | HRS source training, NLSY79 transfer, target development, locked target test |
| MICE with old dummy groups | Fold-fitted median numeric imputation, missing indicators, categorical parent reconstruction and one-hot encoding |
| Separately stored model/imputer | One saved object containing preprocessing, exact feature order, and regressor |
| Existing fitted classifier weights | New Part 2 fits; old classifiers remain in the original project |

Numeric imputation/scaling is fitted only on the relevant training rows. Existing shared dummy groups are reconstructed as one categorical parent; missing parents become an explicit missing category. Raw coded NLSY79 categorical fields are encoded as categories, rather than treating occupation codes as numeric distances. Unseen categories use the fitted encoder's unknown-category behavior. An entirely missing training numeric column retains its position. The first executable version uses this deterministic preprocessing for all learners; MICE is a potential separately documented sensitivity, not silently inherited from the old model.

## Saved artifacts and reuse

Each selected model has `pipeline.joblib` and `metadata.json`, with the feature order, fitted parameters, environment, training-record checksum, development selection score, and save/reload prediction difference. CatBoost and LightGBM also save their native regressor files. Native regressors alone do not include preprocessing; inference should load the complete pipeline.

Every completed stage writes `_SUCCESS.json`, covering its configuration, source-data digest, code digests, environment, and output-file hashes. Repeating the same command reuses a completed stage only if the full contract and artifacts match. A changed configuration needs a new output directory. An interrupted directory is retained for diagnosis; it does not count as a completed stage. Automatic partial-fold resumption is not implemented.

`python -m part2_models predict` reloads a saved model for prediction without retraining. Loading checks the recorded software versions and pipeline hash. Keep the environment lock and the installed project code with the model. This follows the environment requirements in the [scikit-learn persistence documentation](https://scikit-learn.org/stable/model_persistence.html).

## Expected-change model and planning revision

The earlier planning draft treated the endpoint prediction `S_score` as the added HRS feature. That comparison remains implemented as `B_score`. To implement slide 6 explicitly, this revision adds a conditional trajectory of cognitive change and `B_change`, which is the prespecified primary incremental comparison in `experiment_config.json`.

For baseline information X and elapsed years t, the change model is:

`delta(X,t) = beta_t*(t/10) + beta_tt*(t/10)^2 + (t/10)*Z(X)*beta_X`

Z contains training-fitted transformed baseline predictors, including baseline cognition, baseline age, and birth year. The model fits observed follow-up minus baseline cognition with weighted ridge regression. It has no intercept after the time expansion, so delta(X,0)=0. Its endpoint is baseline cognition plus delta(X,t). The penalty is selected by HRS household CV. NLSY79 receives the population conditional prediction; it receives no HRS person-specific random effect.

This is a pair-based longitudinal conditional mean model. Repeated HRS pairs inform the time-dependent change curve. It is not a fitted individual random-effects trajectory, a causal estimate, or proof of individual decline. Future trajectory extensions should be added under separate configurations before test evaluation if they are to participate in the confirmatory comparison.

The endpoint prediction minus baseline (`endpoint_delta`) is also exported. It differs from `S_delta`, which comes from the explicitly fitted change model. Do not interchange these columns in the experiment labels.

## Boundaries

Both distributed ZIP archives remain byte-for-byte unchanged. Unified metadata lives in the repository and refers to those fixed payloads. The project contains source code and aggregate metadata; participant tables and newly fitted artifacts remain in the designated local/cluster data and run directories. Full experiments have not been run as part of this code delivery.
