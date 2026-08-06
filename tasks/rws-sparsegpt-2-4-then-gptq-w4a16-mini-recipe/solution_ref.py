import numpy as np


def _invert_matrix(A):
    n = len(A)
    M = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(A)]
    for i in range(n):
        max_el = abs(M[i][i])
        max_row = i
        for k in range(i + 1, n):
            if abs(M[k][i]) > max_el:
                max_el = abs(M[k][i])
                max_row = k
        if max_row != i:
            M[i], M[max_row] = M[max_row], M[i]
        pivot = M[i][i]
        for j in range(2 * n):
            M[i][j] /= pivot
        for k in range(n):
            if k != i:
                factor = M[k][i]
                for j in range(2 * n):
                    M[k][j] -= factor * M[i][j]
    return [row[n:] for row in M]


def _sparsegpt_2_4(W, X, lam_prune):
    W = np.asarray(W, dtype=np.float64).copy()
    m, n = W.shape
    H = [[0.0] * n for _ in range(n)]
    s_dim = X.shape[1]
    for i in range(n):
        for j in range(n):
            dot = 0.0
            for k in range(s_dim):
                dot += X[i, k] * X[j, k]
            val = 2.0 * dot
            if i == j:
                val += lam_prune
            H[i][j] = val
    Hinv = _invert_matrix(H)

    for r in range(m):
        for start in range(0, n, 4):
            cols = list(range(start, start + 4))
            scores = [(W[r, c] ** 2) / Hinv[c][c] for c in cols]
            keep = set(cols)
            paired = sorted(zip(cols, scores), key=lambda x: x[1])
            for c, _ in paired[:2]:
                keep.remove(c)
            pruned = [c for c in cols if c not in keep]
            for c in pruned:
                old = W[r, c]
                for k in cols:
                    if k in keep:
                        W[r, k] -= old * Hinv[k][c] / Hinv[c][c]
                W[r, c] = 0.0
    return W


def _gptq_quantize(W, X, bits, damp):
    W = np.asarray(W, dtype=np.float64).copy()
    m, n = W.shape
    H = [[0.0] * n for _ in range(n)]
    s_dim = X.shape[1]
    for i in range(n):
        for j in range(n):
            dot = 0.0
            for k in range(s_dim):
                dot += X[i, k] * X[j, k]
            H[i][j] = dot

    diag_sum = 0.0
    for i in range(n):
        diag_sum += H[i][i]
    mean_diag = diag_sum / n

    damp_val = damp * mean_diag
    for i in range(n):
        H[i][i] += damp_val

    Hinv = _invert_matrix(H)

    maxq = (1 << (bits - 1)) - 1
    row_scale = []
    for r in range(m):
        max_val = 0.0
        for c in range(n):
            val = abs(W[r, c])
            if val > max_val:
                max_val = val
        s = max_val / maxq
        if s == 0.0:
            s = 1.0
        row_scale.append(s)

    W_q = np.zeros_like(W)
    for i in range(n):
        q = []
        for r in range(m):
            val = W[r, i] / row_scale[r]
            rounded = round(val)
            clipped = max(-maxq, min(maxq, rounded))
            q_val = clipped * row_scale[r]
            q.append(q_val)
            W_q[r, i] = q_val
        
        err = [q[r] - W[r, i] for r in range(m)]
        if i + 1 < n:
            h_inv_ii = Hinv[i][i]
            coeff = [Hinv[i][j] / h_inv_ii for j in range(i + 1, n)]
            for r in range(m):
                e_r = err[r]
                for idx, j in enumerate(range(i + 1, n)):
                    W[r, j] -= e_r * coeff[idx]
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
