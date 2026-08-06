import numpy as np


def _quantize(a, s):
    rows = a.shape[0]
    cols = a.shape[1] if a.ndim > 1 else 1
    mx = 0.0
    if a.ndim > 1:
        for i in range(rows):
            for j in range(cols):
                val = a[i, j]
                abs_val = val if val >= 0.0 else -val
                if abs_val > mx:
                    mx = abs_val
    else:
        for i in range(rows):
            val = a[i]
            abs_val = val if val >= 0.0 else -val
            if abs_val > mx:
                mx = abs_val

    out = np.zeros(a.shape, dtype=np.float64)
    if mx == 0.0:
        return out

    scale = mx / s

    if a.ndim > 1:
        for i in range(rows):
            for j in range(cols):
                val = a[i, j] / scale
                rnd = round(val)
                if rnd < -s:
                    clipped = -s
                elif rnd > s:
                    clipped = s
                else:
                    clipped = rnd
                out[i, j] = clipped * scale
    else:
        for i in range(rows):
            val = a[i] / scale
            rnd = round(val)
            if rnd < -s:
                clipped = -s
            elif rnd > s:
                clipped = s
            else:
                clipped = rnd
            out[i] = clipped * scale

    return out


def lazy_batch_update(W, X, s, blocksize):
    W_work = np.asarray(W, dtype=np.float64).copy()
    
    m_dim = X.shape[0]
    XXt = np.zeros((m_dim, m_dim), dtype=np.float64)
    for i in range(m_dim):
        for j in range(m_dim):
            acc = 0.0
            for l in range(X.shape[1]):
                acc += X[i, l] * X[j, l]
            XXt[i, j] = acc

    for i in range(m_dim):
        XXt[i, i] += 1e-6

    n = XXt.shape[0]
    M = np.zeros((n, 2 * n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            M[i, j] = XXt[i, j]
        M[i, n + i] = 1.0

    for i in range(n):
        max_val = abs(M[i, i])
        max_row = i
        for k in range(i + 1, n):
            val = abs(M[k, i])
            if val > max_val:
                max_val = val
                max_row = k

        if max_row != i:
            for c in range(2 * n):
                temp = M[i, c]
                M[i, c] = M[max_row, c]
                M[max_row, c] = temp

        inv_pivot = 1.0 / M[i, i]
        for c in range(2 * n):
            M[i, c] *= inv_pivot

        for r in range(n):
            if r != i:
                factor = M[r, i]
                if factor != 0.0:
                    for c in range(2 * n):
                        M[r, c] -= factor * M[i, c]

    h_inv = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            h_inv[i, j] = M[i, n + j]

    n_cols = W_work.shape[1]

    for start in range(0, n_cols, blocksize):
        end = min(n_cols, start + blocksize)

        for j in range(start, end):
            old = W_work[:, j].copy()
            q = _quantize(old.reshape(-1, 1), s).reshape(-1)
            err = old - q
            W_work[:, j] = q

            if j + 1 < n_cols:
                h_jj = h_inv[j, j]
                rows = W_work.shape[0]
                for r in range(rows):
                    err_r_div = err[r] / h_jj
                    for c_idx in range(j + 1, n_cols):
                        W_work[r, c_idx] -= err_r_div * h_inv[j, c_idx]

    return W_work
