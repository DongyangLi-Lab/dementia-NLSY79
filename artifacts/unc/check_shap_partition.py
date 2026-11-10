"""Check the sharded kernel algebra against the original implementation."""
import numpy as np
import pandas as pd
from inference import importance as imp

rng = np.random.default_rng(11)
cols = [f'x{i}' for i in range(8)]
dev = pd.DataFrame(rng.normal(size=(25, 8)), columns=cols)
ext = pd.DataFrame(rng.normal(size=(12, 8)), columns=cols)
groups = {x: [x] for x in cols}
weights = np.arange(1, 9) / 16
imp._predict = lambda model, X: 1 / (1 + np.exp(-(X.to_numpy() @ weights)))
names, actual = imp.kernel_shap(None, dev, ext, groups, n_explain=6, n_coalitions=20,
                                n_background=3, seed=2026, chunk=60)
rng = np.random.default_rng(2026)
xe = ext.to_numpy(np.float32)[rng.choice(len(ext), 6, replace=False)]
bg = dev.to_numpy(np.float32)[rng.choice(len(dev), 3, replace=False)]
M = len(names); sizes = np.arange(1, M); w = (M - 1) / (sizes * (M - sizes))
Z = np.zeros((20, M), bool)
for i in range(0, 20, 2):
    Z[i, rng.choice(M, rng.choice(sizes, p=w / w.sum()), replace=False)] = True
    Z[i + 1] = ~Z[i]
mask = Z.copy()
logit = lambda p: np.log(np.clip(p, 1e-6, 1-1e-6) / (1-np.clip(p, 1e-6, 1-1e-6)))
predict = lambda a: imp._predict(None, pd.DataFrame(a, columns=cols))
v0 = logit(predict(bg).mean()); v1 = logit(predict(xe))
z = Z.astype(float); D = z[:, :-1] - z[:, [-1]]
chunks = []
for start, end in ((0, 3), (3, 6)):
    V = np.empty((20, end-start))
    for k, s in enumerate(range(start, end)):
        A = np.where(mask[:, None, :], xe[s][None, None, :], bg[None, :, :])
        V[:, k] = logit(predict(A.reshape(-1, len(cols))).reshape(20, 3).mean(axis=1))
    Y = (V-v0) - z[:, [-1]] * (v1[start:end]-v0)[None, :]
    phi = np.linalg.lstsq(D, Y, rcond=None)[0].T
    chunks.append(np.column_stack([phi, (v1[start:end]-v0)-phi.sum(axis=1)]))
assembled = np.concatenate(chunks)
assert np.allclose(actual, assembled, rtol=1e-10, atol=1e-10)
print('SHAP shard equivalence max error', np.max(np.abs(actual-assembled)))
