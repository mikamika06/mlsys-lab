import numpy as np


def moe_dispatch_combine(X: np.ndarray, expert_idx: np.ndarray,
                          gate_weight: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    Dispatch tokens to their assigned expert, apply that expert's linear
    transform, and combine results back into the original token order,
    scaled by the per-token gate weight.
    """
    X = np.asarray(X, dtype=np.float64)
    expert_idx = np.asarray(expert_idx)
    gate_weight = np.asarray(gate_weight, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    n, d = X.shape

    # --- dispatch: sort tokens by expert so each expert's tokens are contiguous
    order = np.argsort(expert_idx, kind="stable")
    sorted_idx = expert_idx[order]
    X_sorted = X[order]

    # --- expert compute: one batched matmul per contiguous expert group
    out_sorted = np.empty_like(X_sorted)
    start = 0
    while start < n:
        e = int(sorted_idx[start])
        end = start
        while end < n and sorted_idx[end] == e:
            end += 1
        out_sorted[start:end] = X_sorted[start:end] @ W[e]
        start = end

    # --- combine: scatter back to original order, then scale by gate weight
    Y = np.empty_like(X)
    Y[order] = out_sorted
    Y = Y * gate_weight[:, None]
    return Y
