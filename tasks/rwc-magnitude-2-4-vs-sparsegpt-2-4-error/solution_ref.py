import math
import numpy as np

LAMBDA = 1e-4


def _invert_matrix(mat):
    n = len(mat)
    M = [[float(mat[i][j]) for j in range(n)] + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for i in range(n):
        max_val = abs(M[i][i])
        max_row = i
        for k in range(i + 1, n):
            if abs(M[k][i]) > max_val:
                max_val = abs(M[k][i])
                max_row = k
        if max_row != i:
            M[i], M[max_row] = M[max_row], M[i]
        pivot = M[i][i]
        inv_pivot = 1.0 / pivot
        for j in range(2 * n):
            M[i][j] *= inv_pivot
        for k in range(n):
            if k != i:
                factor = M[k][i]
                for j in range(2 * n):
                    M[k][j] -= factor * M[i][j]
    return [[M[i][j + n] for j in range(n)] for i in range(n)]


def _magnitude_prune_2_4(W: np.ndarray) -> np.ndarray:
    m, n = W.shape
    out = W.copy()
    for r in range(m):
        for start in range(0, n, 4):
            cols = list(range(start, start + 4))
            order = sorted(range(4), key=lambda i: abs(W[r, cols[i]]))
            for c in [cols[order[0]], cols[order[1]]]:
                out[r, c] = 0.0
    return out


def _sparsegpt_2_4(W: np.ndarray, X: np.ndarray) -> np.ndarray:
    s = X.shape[0]
    in_dim = X.shape[1]
    H_list = [[0.0 for _ in range(in_dim)] for _ in range(in_dim)]
    for i in range(in_dim):
        for j in range(in_dim):
            acc = 0.0
            for k in range(s):
                acc += X[k, i] * X[k, j]
            val = acc / s
            if i == j:
                val += LAMBDA
            H_list[i][j] = val

    Hinv = np.array(_invert_matrix(H_list), dtype=np.float64)

    mask = np.zeros((W.shape[0], W.shape[1]), dtype=np.int64)
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

    s = X.shape[0]
    in_dim = X.shape[1]
    out_dim = W.shape[0]

    Y_true_list = [[0.0 for _ in range(out_dim)] for _ in range(s)]
    for i in range(s):
        for j in range(out_dim):
            acc = 0.0
            for l in range(in_dim):
                acc += X[i, l] * W[j, l]
            Y_true_list[i][j] = acc
    Y_true = np.array(Y_true_list, dtype=np.float64)

    W_mag = _magnitude_prune_2_4(W)
    Y_mag_list = [[0.0 for _ in range(out_dim)] for _ in range(s)]
    for i in range(s):
        for j in range(out_dim):
            acc = 0.0
            for l in range(in_dim):
                acc += X[i, l] * W_mag[j, l]
            Y_mag_list[i][j] = acc
    Y_mag = np.array(Y_mag_list, dtype=np.float64)

    num_mag_sq = 0.0
    den_sq = 0.0
    for i in range(s):
        for j in range(out_dim):
            diff = Y_mag[i, j] - Y_true[i, j]
            num_mag_sq += diff * diff
            t = Y_true[i, j]
            den_sq += t * t
    err_mag = float(math.sqrt(num_mag_sq) / math.sqrt(den_sq))

    W_sp = _sparsegpt_2_4(W, X)
    Y_sp_list = [[0.0 for _ in range(out_dim)] for _ in range(s)]
    for i in range(s):
        for j in range(out_dim):
            acc = 0.0
            for l in range(in_dim):
                acc += X[i, l] * W_sp[j, l]
            Y_sp_list[i][j] = acc
    Y_sp = np.array(Y_sp_list, dtype=np.float64)

    num_sp_sq = 0.0
    for i in range(s):
        for j in range(out_dim):
            diff = Y_sp[i, j] - Y_true[i, j]
            num_sp_sq += diff * diff
    err_sp = float(math.sqrt(num_sp_sq) / math.sqrt(den_sq))

    reduction = 1.0 - err_sp / err_mag
    return err_mag, err_sp, reduction
