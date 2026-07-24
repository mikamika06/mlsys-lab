import numpy as np

LAMBDA = 1e-4


def _magnitude_prune_2_4(W: np.ndarray) -> np.ndarray:
    m, n = W.shape
    out = W.copy()
    for r in range(m):
        for start in range(0, n, 4):
            cols = list(range(start, start + 4))
            block = W[r, cols]
            order = np.argsort(np.abs(block))
            for c in [cols[order[0]], cols[order[1]]]:
                out[r, c] = 0.0
    return out


def _sparsegpt_2_4(W: np.ndarray, X: np.ndarray) -> np.ndarray:
    s = X.shape[0]
    H = (X.T @ X) / s + LAMBDA * np.eye(X.shape[1])
    Hinv = np.linalg.inv(H)

    mask = np.zeros_like(W, dtype=np.int64)
    out = W.copy()

    for r in range(W.shape[0]):
        for start in range(0, W.shape[1], 4):
            cols = list(range(start, start + 4))
            scores = [(W[r, c] ** 2) / Hinv[c, c] for c in cols]
            keep = set(cols)
            for c, _ in sorted(zip(cols, scores), key=lambda x: x[1])[:2]:
                keep.remove(c)
            for c in keep:
                mask[r, c] = 1

            pruned = [c for c in cols if mask[r, c] == 0]
            for c in pruned:
                old = out[r, c]
                for k in cols:
                    if mask[r, k] == 1:
                        out[r, k] -= old * Hinv[k, c] / Hinv[c, c]
                out[r, c] = 0.0

    return out


def compare_magnitude_vs_sparsegpt_2_4(W: np.ndarray, X: np.ndarray):
    """
    Compare naive magnitude 2:4 pruning against Hessian-aware SparseGPT
    2:4 pruning, by the relative Frobenius-norm output error each
    introduces on the linear layer Y = X @ W.T.

    Returns (err_magnitude, err_sparsegpt, reduction), with
    reduction = 1 - err_sparsegpt / err_magnitude.
    """
    W = np.asarray(W, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    Y_true = X @ W.T

    W_mag = _magnitude_prune_2_4(W)
    err_mag = float(np.linalg.norm(X @ W_mag.T - Y_true) / np.linalg.norm(Y_true))

    W_sp = _sparsegpt_2_4(W, X)
    err_sp = float(np.linalg.norm(X @ W_sp.T - Y_true) / np.linalg.norm(Y_true))

    reduction = 1.0 - err_sp / err_mag
    return err_mag, err_sp, reduction
