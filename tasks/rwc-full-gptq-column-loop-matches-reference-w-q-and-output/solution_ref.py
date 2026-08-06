import math
import numpy as np


def gptq_quantize(W, X, bits=3, group_size=2, damp=0.01):
    W_work = np.asarray(W, dtype=np.float64).copy()
    m, n = W_work.shape

    H = [[0.0] * n for _ in range(n)]
    X_rows = X.tolist()
    for r1 in range(n):
        for r2 in range(n):
            acc = 0.0
            for k in range(len(X_rows[0])):
                acc += X_rows[r1][k] * X_rows[r2][k]
            H[r1][r2] = acc

    diag_sum = 0.0
    for i in range(n):
        diag_sum += H[i][i]
    mean_diag = diag_sum / n

    for i in range(n):
        H[i][i] += damp * mean_diag

    aug = [row[:] + [1.0 if r == c else 0.0 for c in range(n)] for r, row in enumerate(H)]
    for col in range(n):
        pivot_row = col
        for r in range(col + 1, n):
            if abs(aug[r][col]) > abs(aug[pivot_row][col]):
                pivot_row = r
        aug[col], aug[pivot_row] = aug[pivot_row], aug[col]

        factor = aug[col][col]
        for c in range(2 * n):
            aug[col][c] /= factor

        for r in range(n):
            if r != col:
                f = aug[r][col]
                for c in range(2 * n):
                    aug[r][c] -= f * aug[col][c]

    Hinv = [row[n:] for row in aug]

    W_q = np.zeros_like(W_work)
    maxq = (1 << (bits - 1)) - 1

    scales = {}
    for start in range(0, n, group_size):
        end = min(n, start + group_size)
        scale = np.zeros(m, dtype=np.float64)
        for row_idx in range(m):
            max_val = 0.0
            for col_idx in range(start, end):
                val = abs(W_work[row_idx, col_idx])
                if val > max_val:
                    max_val = val
            s = max_val / maxq
            if s == 0.0:
                s = 1.0
            scale[row_idx] = s
        scales[start] = scale

    for i in range(n):
        start = (i // group_size) * group_size
        scale = scales[start]
        
        q = np.zeros(m, dtype=np.float64)
        err = np.zeros(m, dtype=np.float64)
        for row_idx in range(m):
            val = W_work[row_idx, i] / scale[row_idx]
            rounded = round(val)
            if rounded < -maxq:
                clipped = -maxq
            elif rounded > maxq:
                clipped = maxq
            else:
                clipped = rounded
            q_val = clipped * scale[row_idx]
            q[row_idx] = q_val
            W_q[row_idx, i] = q_val
            err[row_idx] = q_val - W_work[row_idx, i]

        if i + 1 < n:
            hinv_ii = Hinv[i][i]
            coeff = [Hinv[i][c] / hinv_ii for c in range(i + 1, n)]
            for row_idx in range(m):
                e = err[row_idx]
                for idx_c, c_col in enumerate(range(i + 1, n)):
                    W_work[row_idx, c_col] -= e * coeff[idx_c]

    W_q_list = W_q.tolist()
    X_list = X.tolist()
    res_rows = len(W_q_list)
    res_cols = len(X_list[0])
    common_dim = len(X_list)
    
    W_q_X = np.zeros((res_rows, res_cols), dtype=np.float64)
    for r in range(res_rows):
        for c in range(res_cols):
            acc = 0.0
            for k in range(common_dim):
                acc += W_q_list[r][k] * X_list[k][c]
            W_q_X[r, c] = acc

    return W_q, W_q_X
