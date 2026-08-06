import numpy as np


def _quantize_column(x):
    x = np.asarray(x, dtype=np.float64)
    max_val = 0.0
    for i in range(len(x)):
        val = x[i]
        abs_val = val if val >= 0 else -val
        if abs_val > max_val:
            max_val = abs_val
    scale = max_val / 7.0
    if scale == 0:
        return np.zeros_like(x)
    res = np.zeros_like(x)
    for i in range(len(x)):
        val = x[i] / scale
        r = round(val)
        if r < -8:
            r = -8.0
        elif r > 7:
            r = 7.0
        else:
            r = float(r)
        res[i] = r * scale
    return res


def _invert_matrix(A):
    n = A.shape[0]
    M = []
    for i in range(n):
        row = list(A[i]) + [1.0 if i == j else 0.0 for j in range(n)]
        M.append(row)
    for i in range(n):
        max_row = i
        max_val = M[i][i] if M[i][i] >= 0 else -M[i][i]
        for r in range(i + 1, n):
            val = M[r][i] if M[r][i] >= 0 else -M[r][i]
            if val > max_val:
                max_val = val
                max_row = r
        if max_row != i:
            M[i], M[max_row] = M[max_row], M[i]
        pivot = M[i][i]
        for c in range(2 * n):
            M[i][c] /= pivot
        for r in range(n):
            if r != i:
                factor = M[r][i]
                if factor != 0:
                    for c in range(2 * n):
                        M[r][c] -= factor * M[i][c]
    inv_A = np.zeros_like(A)
    for i in range(n):
        for j in range(n):
            inv_A[i, j] = M[i][n + j]
    return inv_A


def gptq_act_order(W: np.ndarray, H: np.ndarray):
    n = W.shape[1]
    diag_vals = []
    for i in range(n):
        diag_vals.append((-H[i, i], i))
    sorted_diag = sorted(diag_vals, key=lambda item: item[0])
    perm = np.array([item[1] for item in sorted_diag], dtype=np.int64)
    inv_h = _invert_matrix(H)

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
