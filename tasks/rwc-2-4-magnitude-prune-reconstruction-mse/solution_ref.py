import numpy as np

def magnitude_prune_mse(W):
    """
    Compute the mean squared error between W and its 2:4 magnitude‑pruned version.
    """
    W = np.asarray(W, dtype=np.float64)
    n_rows, n_cols = W.shape
    pruned = np.zeros_like(W)
    for i in range(n_rows):
        row = W[i]
        for j in range(0, n_cols, 4):
            group = row[j:j+4]
            if group.size == 0:
                continue
            idx = np.argpartition(np.abs(group), -2)[-2:]
            mask = np.zeros_like(group, dtype=bool)
            mask[idx] = True
            pruned[i,j:j+4][mask] = group[mask]
    mse = np.mean((W - pruned) ** 2)
    return float(mse)
