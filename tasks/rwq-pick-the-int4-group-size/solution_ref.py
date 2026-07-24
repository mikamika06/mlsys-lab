import numpy as np


def _group_quant_mse(W: np.ndarray, group_size: int, bits: int) -> float:
    qmax = (1 << (bits - 1)) - 1
    n = W.shape[0]
    blocks = W.reshape(n // group_size, group_size)
    amax = np.max(np.abs(blocks), axis=1, keepdims=True)
    scale = np.where(amax > 0, amax / qmax, 1.0)
    codes = np.clip(np.round(blocks / scale), -qmax, qmax)
    recon = codes * scale
    return float(np.mean((recon - blocks) ** 2))


def pick_int4_group_size(W: np.ndarray, group_sizes=(32, 64, 128, 256),
                          bits: int = 4, lam: float = 0.02):
    """Pick the group_size minimizing mse(gs) + lam * (16.0 / gs).

    Returns (best_group_size, best_cost, costs), costs aligned with
    `group_sizes` order.
    """
    W = np.asarray(W, dtype=np.float64)
    costs = np.zeros(len(group_sizes), dtype=np.float64)
    for i, gs in enumerate(group_sizes):
        mse = _group_quant_mse(W, gs, bits)
        overhead = 16.0 / gs
        costs[i] = mse + lam * overhead

    best_idx = int(np.argmin(costs))
    return int(group_sizes[best_idx]), float(costs[best_idx]), costs
