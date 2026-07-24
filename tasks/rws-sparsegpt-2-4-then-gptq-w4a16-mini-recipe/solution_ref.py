import numpy as np


def _sparsegpt_2_4(W, X, lam_prune):
    W = np.asarray(W, dtype=np.float64).copy()
    m, n = W.shape
    H = 2.0 * X @ X.T + lam_prune * np.eye(n)
    Hinv = np.linalg.inv(H)

    for r in range(m):
        for start in range(0, n, 4):
            cols = list(range(start, start + 4))
            scores = [(W[r, c] ** 2) / Hinv[c, c] for c in cols]
            keep = set(cols)
            for c, _ in sorted(zip(cols, scores), key=lambda x: x[1])[:2]:
                keep.remove(c)
            pruned = [c for c in cols if c not in keep]
            for c in pruned:
                old = W[r, c]
                for k in cols:
                    if k in keep:
                        W[r, k] -= old * Hinv[k, c] / Hinv[c, c]
                W[r, c] = 0.0
    return W


def _gptq_quantize(W, X, bits, damp):
    W = np.asarray(W, dtype=np.float64).copy()
    m, n = W.shape
    H = X @ X.T
    H = H + np.eye(n) * damp * np.mean(np.diag(H))
    Hinv = np.linalg.inv(H)

    maxq = (1 << (bits - 1)) - 1
    row_scale = np.max(np.abs(W), axis=1) / maxq
    row_scale = np.where(row_scale == 0.0, 1.0, row_scale)

    W_q = np.zeros_like(W)
    for i in range(n):
        q = np.clip(np.round(W[:, i] / row_scale), -maxq, maxq) * row_scale
        W_q[:, i] = q
        err = q - W[:, i]
        if i + 1 < n:
            coeff = Hinv[i, i + 1:] / Hinv[i, i]
            W[:, i + 1:] -= np.outer(err, coeff)
    return W_q


def sparsegpt_then_gptq(
    W: np.ndarray,
    X: np.ndarray,
    bits: int = 4,
    lam_prune: float = 1e-2,
    damp: float = 1e-2,
) -> np.ndarray:
    """
    Two-stage compression recipe:

    1. SparseGPT 2:4 structured pruning: within every group of 4
       consecutive columns of each row, prune the 2 lowest-saliency
       weights (saliency = w^2 / diag(H_prune^-1)) and compensate the
       2 surviving weights via the inverse Hessian.
    2. GPTQ Hessian-ordered per-column int-`bits` quantization of the
       pruned ("surviving") weights, using a per-row symmetric scale
       and propagating rounding error to not-yet-quantized columns via
       a second (independently damped) inverse Hessian.

    W: (m, n) float64 weight matrix, n divisible by 4.
    X: (n, s) float64 calibration activations (rows = input features,
       columns = samples).

    Returns W_hat: (m, n) float64 pruned + quantized reconstruction.
    """
    W = np.asarray(W, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    W_sparse = _sparsegpt_2_4(W, X, lam_prune)
    return _gptq_quantize(W_sparse, X, bits, damp)
