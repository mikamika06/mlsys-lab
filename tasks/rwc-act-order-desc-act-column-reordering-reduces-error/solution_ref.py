import numpy as np


def _quantize_column(x):
    x = np.asarray(x, dtype=np.float64)
    scale = np.max(np.abs(x)) / 7.0
    if scale == 0:
        return np.zeros_like(x)
    return np.clip(np.round(x / scale), -8, 7) * scale


def gptq_act_order(W: np.ndarray, H: np.ndarray):
    n = W.shape[1]
    perm = np.argsort(-np.diag(H), kind="stable")
    inv_h = np.linalg.inv(H)

    work = W[:, perm].copy()
    out = np.zeros_like(work)

    for j in range(n):
        q = _quantize_column(work[:, j])
        err = work[:, j] - q
        out[:, j] = q
        if j + 1 < n:
            for k in range(j + 1, n):
                work[:, k] -= err * (
                    inv_h[perm[j], perm[k]] / inv_h[perm[j], perm[j]]
                )

    restored = np.zeros_like(out)
    restored[:, perm] = out
    return perm, restored
