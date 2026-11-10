"""Optional batched GAT inference without changing the fitted graph protocol.

This separate implementation leaves existing benchmark source stamps intact.
"""
import numpy as np
import torch
import torch.nn.functional as F


def predict_proba(model, X, batch_size=256):
    """Predict independently against the fitted reference graph, in larger batches."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    X_t = torch.tensor(model._scale(X).to_numpy(dtype=np.float32))
    device = "cuda" if model.cuda_ else "cpu"
    reference = model.reference_.to(device)
    n_nodes = len(reference) + 1
    out = np.empty(len(X_t), dtype=np.float64)
    model.model.eval()
    with torch.no_grad():
        ref_sim = torch.abs(F.cosine_similarity(reference[:, None, :], reference[None, :, :], dim=-1))
        for start in range(0, len(X_t), batch_size):
            chunk = X_t[start:start + batch_size].to(device)
            k = len(chunk)
            graphs = torch.cat([reference.unsqueeze(0).expand(k, -1, -1), chunk.unsqueeze(1)], dim=1)
            sim = torch.empty((k, n_nodes, n_nodes), device=device, dtype=graphs.dtype)
            sim[:, :-1, :-1] = ref_sim
            query_sim = torch.abs(F.cosine_similarity(chunk[:, None, :], reference[None, :, :], dim=-1))
            sim[:, -1, :-1] = query_sim
            sim[:, :-1, -1] = query_sim
            sim[:, -1, -1] = torch.abs(F.cosine_similarity(chunk, chunk, dim=-1))
            eps = torch.quantile(sim.reshape(k, -1), model.alpha, dim=1, interpolation="nearest")
            g, row, col = torch.where(sim >= eps[:, None, None])
            edges = torch.stack([row + g * n_nodes, col + g * n_nodes])
            pred = model.model(graphs.reshape(k * n_nodes, -1), edges)
            out[start:start + k] = pred.reshape(k, n_nodes)[:, -1].double().cpu().numpy()
    return np.stack([1 - out, out], 1)
