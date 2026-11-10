"""Feature importance on held-out data: the 2010 temporal-validation cohort.

Load each frozen temporal checkpoint and evaluate it on the imputed 2010 cohort.

- Permutation importance: for each of the 60 predictors, the drop in AUPRC when that predictor is randomly permuted
  in the 2010 data (one-hot groups and functional items with their "don't do" indicator are permuted jointly),
  averaged over N_PERMUTATIONS permutations. The model is not refitted. 95% CIs: household cluster bootstrap of the
  importance, recomputed on each bootstrap sample from the stored predictions.
- SHAP: exact SHAP values for the models where they can be computed exactly: LightGBM (pred_contrib) and CatBoost
  (ShapValues) on the log-odds scale; the decision tree (tree SHAP, shap.TreeExplainer, predicted probability);
  logistic regression (linear SHAP: coefficient x (value - development mean)); and Gaussian naive Bayes, whose
  log-odds are a sum of one term per input, g_j(x_j) = log N(x_j; class 1) - log N(x_j; class 0), so that its exact
  SHAP value is g_j(x_j) minus the development mean of g_j. Values are summed within each predictor group; importance
  is the mean absolute SHAP value. Rank agreement with permutation importance is reported (Spearman).
- Approximate SHAP (--kernel-shap) for RealMLP, TabICL, and GAT, which have no exact method: KernelSHAP over the 60
  predictor groups on the log-odds scale, for a random sample of the 2010 cohort against a random background sample
  of the development cohort (KS_* settings). The same settings are applied to LightGBM and logistic regression, and
  their approximate ranks are compared with the exact ones to check the approximation.

Importance describes how much a model's discrimination depends on a predictor given the other predictors; it does
not indicate a causal or modifiable effect, and correlated predictors share importance.

Usage: python importance.py [--horizon 6] [--models LR LightGBM ...] [--kernel-shap]
"""
import argparse
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score

from artifact_store import job_dir, load_model, save_receipt, verified, write_file
from inference import evaluation
from inference.scoring import predict as _predict
from model.specs import MODELS
from train.cohorts import ARTIFACTS, atomic_csv, load_cohorts, sha256
from train.imputation import cache_stamp, category_groups, read_cache

N_PERMUTATIONS = 5
N_BOOT = 500
SHAP_MODELS = ('LR', 'GNB', 'DT', 'LightGBM', 'CatBoost')
KERNEL_SHAP_MODELS = ('RealMLP', 'TabICL', 'GAT')  # no exact method: approximate KernelSHAP
KS_EXPLAIN, KS_COALITIONS, KS_BACKGROUND, KS_CHUNK = 500, 800, 10, 8000  # chunk: rows per predict call (TabICL GPU memory)


def predictor_groups(predictors):
    """Variable name -> model-input columns (60 groups for the 94 inputs)."""
    groups = {}
    for name, cols in category_groups(predictors).items():
        groups[name[:-len('_item')] if name.endswith('_item') else name] = cols
    grouped = {c for cols in groups.values() for c in cols}
    for c in predictors:
        if c not in grouped:
            groups[c] = [c]
    return groups


def load_temporal_model(model_name, horizon):
    _, ext, predictors = load_cohorts(horizon)
    cache = read_cache(horizon, "final", "external", cache_stamp(horizon, "final"))
    model = load_model(job_dir("checkpoints", horizon, "final", model_name) / "temporal.pkl")
    return model, cache["X_dev"], cache["X_ext"].astype(np.float32), ext.y.to_numpy(), ext.hhidpn.to_numpy(), predictors


def permutation_importance(model_name, horizon, target, stamp, n_perm=N_PERMUTATIONS, n_boot=N_BOOT, seed=2026):
    model, X_dev, X_ext, y, hhidpn, P = load_temporal_model(model_name, horizon)
    groups = predictor_groups(P)
    base = _predict(model, X_ext)
    saved = evaluation.load(horizon, model_name, 'temporal')['p'].to_numpy()
    if not np.allclose(base, saved, rtol=1e-5, atol=1e-7):
        raise ValueError(f"{model_name}: frozen checkpoint did not reproduce held-out predictions")
    permuted = {}
    t0 = time.time()
    for i, (name, cols) in enumerate(groups.items()):
        raw = target / "permutations" / f"{i:02d}.npy"
        receipt = raw.with_suffix(".done.json")
        if not verified(receipt, stamp, (raw,)):
            rng = np.random.default_rng(np.random.SeedSequence([seed, i]))
            probabilities = []
            for _ in range(n_perm):
                Xp = X_ext.copy()
                Xp[cols] = X_ext[cols].to_numpy()[rng.permutation(len(Xp))]
                probabilities.append(_predict(model, Xp))
            def save(p):
                with p.open("wb") as stream:
                    np.save(stream, np.asarray(probabilities))
            write_file(raw, save)
            save_receipt(receipt, stamp, (raw,))
        permuted[name] = np.load(raw)
        print(f"{model_name} {name} permutation predictions ready", flush=True)
    seconds = time.time() - t0
    base_auprc = average_precision_score(y, base)
    point = {g: base_auprc - np.mean([average_precision_score(y, p) for p in ps]) for g, ps in permuted.items()}
    boot = {g: [] for g in groups}
    for idx in evaluation.cluster_bootstrap_indices(hhidpn, n_boot, seed=seed):
        yb = y[idx]
        b0 = average_precision_score(yb, base[idx])
        for g, ps in permuted.items():
            boot[g].append(b0 - np.mean([average_precision_score(yb, p[idx]) for p in ps]))
    rows = []
    for g in groups:
        lo, hi = np.percentile(boot[g], [2.5, 97.5])
        rows.append({'horizon': horizon, 'model': model_name, 'variable': g, 'importance_AUPRC_drop': point[g],
                     'ci_low': lo, 'ci_high': hi, 'n_inputs': len(groups[g])})
    df = pd.DataFrame(rows)
    df['rank'] = df['importance_AUPRC_drop'].rank(ascending=False, method='min').astype(int)
    meta = {'horizon': horizon, 'model': model_name, 'base_AUPRC': base_auprc, 'permutation_seconds': seconds,
            'max_abs_diff_vs_saved_predictions': float(np.max(np.abs(base - saved)))}
    return df.sort_values('rank'), meta, model, X_dev, X_ext, groups


def shap_importance(model_name, model, X_dev, X_ext, groups, horizon):
    if model_name == 'LightGBM':
        phi = model.predict(X_ext, pred_contrib=True)[:, :-1]
    elif model_name == 'CatBoost':
        from catboost import Pool
        phi = model.get_feature_importance(Pool(X_ext), type='ShapValues')[:, :-1]
    elif model_name == 'LR':
        phi = (X_ext.to_numpy() - X_dev.to_numpy(dtype=np.float32).mean(axis=0)) * model.coef_[0]
    elif model_name == 'GNB':
        g_ext, g_dev = _gnb_terms(model, X_ext.to_numpy()), _gnb_terms(model, X_dev.to_numpy(dtype=np.float32))
        phi = g_ext - g_dev.mean(axis=0)
    elif model_name == 'DT':
        import shap
        phi = np.asarray(shap.TreeExplainer(model).shap_values(X_ext.to_numpy(), check_additivity=True))
        phi = phi[..., 1] if phi.ndim == 3 else phi[1]  # class-1 probability
    else:
        raise ValueError(model_name)
    phi = pd.DataFrame(phi, columns=X_ext.columns)
    rows = [{'horizon': horizon, 'model': model_name, 'variable': g,
             'mean_abs_SHAP': float(np.abs(phi[cols].sum(axis=1)).mean())} for g, cols in groups.items()]
    df = pd.DataFrame(rows)
    df['rank'] = df['mean_abs_SHAP'].rank(ascending=False, method='min').astype(int)
    return df.sort_values('rank')


def kernel_shap(model, X_dev, X_ext, groups, n_explain=KS_EXPLAIN, n_coalitions=KS_COALITIONS,
                n_background=KS_BACKGROUND, seed=2026, chunk=KS_CHUNK):
    """Approximate SHAP values (KernelSHAP) on the log-odds scale for predictor groups, for models without an exact
    method. Explains a random sample of the 2010 cohort against a random background sample of the development cohort.
    Coalitions are drawn from the Shapley kernel distribution in complementary pairs, and the values are the
    least-squares solution under the efficiency constraint (Lundberg and Lee 2017). Returns (group names, SHAP values
    of shape (n_explain, n_groups)); the sample and background depend only on the seed, so every model is explained
    on the same participants."""
    rng = np.random.default_rng(seed)
    names = list(groups)
    cols = list(X_ext.columns)
    M = len(names)
    Xe = X_ext.to_numpy(np.float32)[rng.choice(len(X_ext), n_explain, replace=False)]
    bg = X_dev.to_numpy(np.float32)[rng.choice(len(X_dev), n_background, replace=False)]
    sizes = np.arange(1, M)
    w = (M - 1) / (sizes * (M - sizes))
    Z = np.zeros((n_coalitions, M), bool)
    for i in range(0, n_coalitions, 2):
        Z[i, rng.choice(M, rng.choice(sizes, p=w / w.sum()), replace=False)] = True
        Z[i + 1] = ~Z[i]
    colmask = np.zeros((n_coalitions, len(cols)), bool)
    for g, name in enumerate(names):
        colmask[:, [cols.index(c) for c in groups[name]]] = Z[:, [g]]
    logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    predict = lambda A: _predict(model, pd.DataFrame(A, columns=cols))
    v0 = logit(predict(bg).mean())
    v1 = logit(predict(Xe))
    per_call = max(1, chunk // (n_coalitions * n_background))
    V = np.empty((n_coalitions, n_explain))
    for s in range(0, n_explain, per_call):
        xs = Xe[s:s + per_call]
        A = np.where(colmask[None, :, None, :], xs[:, None, None, :], bg[None, None, :, :])
        p = predict(A.reshape(-1, len(cols))).reshape(len(xs), n_coalitions, n_background)
        V[:, s:s + len(xs)] = logit(p.mean(axis=2)).T
    # efficiency constraint sum(phi) = v1 - v0: eliminate the last group and solve the remaining least squares
    Zf = Z.astype(float)
    D = Zf[:, :-1] - Zf[:, [-1]]
    Y = (V - v0) - Zf[:, [-1]] * (v1 - v0)[None, :]
    phi = np.linalg.lstsq(D, Y, rcond=None)[0].T
    phi = np.column_stack([phi, (v1 - v0) - phi.sum(axis=1)])
    return names, phi


def _gnb_terms(model, X):
    """Per-input log-likelihood ratio terms of Gaussian naive Bayes; their sum plus the log prior ratio is the
    log-odds of class 1."""
    X = np.asarray(X, float)
    mu, var = model.theta_, model.var_
    logpdf = lambda c: -0.5 * (np.log(2 * np.pi * var[c]) + (X - mu[c]) ** 2 / var[c])
    return logpdf(1) - logpdf(0)


def run_model(model, horizon=6):
    target = job_dir("importance", horizon, "final", model)
    pred = job_dir("predictions", horizon, "final", model) / "inference.done.json"
    stamp = {"inference": sha256(pred), "implementation": sha256(Path(__file__))}
    names = ["permutation", "meta", "shap", "rank_agreement"]
    if model in ("LightGBM", "LR"):
        names.append("kernel_shap_check")
    files = [target / f"{name}.csv" for name in names]
    done = target / "importance.done.json"
    if verified(done, stamp, files):
        return
    perm, meta, estimator, X_dev, X_ext, groups = permutation_importance(model, horizon, target, stamp)
    exact = model in SHAP_MODELS
    method = "exact" if exact else "KernelSHAP (approximate)"
    if exact:
        shap = shap_importance(model, estimator, X_dev, X_ext, groups, horizon)
    if not exact or model in ("LightGBM", "LR"):
        groups_order, phi = kernel_shap(estimator, X_dev, X_ext, groups)
        approximate = pd.DataFrame({"horizon": horizon, "model": model, "variable": groups_order,
                                    "mean_abs_SHAP": np.abs(phi).mean(axis=0)})
        approximate["rank"] = approximate["mean_abs_SHAP"].rank(ascending=False, method="min").astype(int)
        approximate = approximate.sort_values("rank")
        with (target / "kernel_shap_values.npz").open("wb") as stream:
            np.savez_compressed(stream, groups=groups_order, values=phi)
        if not exact:
            shap = approximate
        else:
            check = shap.merge(approximate, on="variable", suffixes=("_exact", "_kernel"))
            rho = stats.spearmanr(check.mean_abs_SHAP_exact, check.mean_abs_SHAP_kernel).correlation
            top = len(set(shap.variable.iloc[:10]) & set(approximate.variable.iloc[:10]))
            atomic_csv(pd.DataFrame([{"horizon": horizon, "model": model,
                                      "spearman_rank_correlation": rho, "shared_top10": top}]), files[4])
    paired = perm.merge(shap, on=["horizon", "model", "variable"])
    agree = {"horizon": horizon, "model": model, "method": method,
             "spearman_rank_correlation": stats.spearmanr(paired.importance_AUPRC_drop, paired.mean_abs_SHAP).correlation,
             "shared_top10": len(set(perm.variable.iloc[:10]) & set(shap.variable.iloc[:10]))}
    for frame, path in zip((perm, pd.DataFrame([meta]), shap.assign(method=method), pd.DataFrame([agree])), files):
        atomic_csv(frame, path)
    save_receipt(done, stamp, files)


def merge(horizon=6):
    names = ("permutation", "meta", "shap", "rank_agreement", "kernel_shap_check")
    for name in names:
        models = ("LR", "LightGBM") if name == "kernel_shap_check" else MODELS
        frames = []
        for model in models:
            target = job_dir("importance", horizon, "final", model)
            if not (target / "importance.done.json").exists():
                raise ValueError(f"Incomplete importance for {model}")
            frames.append(pd.read_csv(target / f"{name}.csv"))
        atomic_csv(pd.concat(frames, ignore_index=True), evaluation.TABLES / f"importance_{name}_h{horizon}.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--model", choices=MODELS)
    action.add_argument("--merge", action="store_true")
    args = parser.parse_args()
    if args.merge:
        merge()
    else:
        run_model(args.model)
