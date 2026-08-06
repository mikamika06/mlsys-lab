import numpy as np
import math


def _xt_x(X):
    m = X.shape[0]
    d_in = X.shape[1]
    H = np.zeros((d_in, d_in), dtype=np.float64)
    for i in range(d_in):
        for j in range(d_in):
            s = 0.0
            for p in range(m):
                s += X[p, i] * X[p, j]
            H[i, j] = s
    return H


def _mean_diag(H):
    n = H.shape[0]
    s = 0.0
    for i in range(n):
        s += H[i, i]
    return s / n


def _eye(n):
    E = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        E[i, i] = 1.0
    return E


def _inv(A):
    n = A.shape[0]
    M = np.zeros((n, 2 * n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            M[i, j] = A[i, j]
        M[i, n + i] = 1.0

    for i in range(n):
        max_val = math.fabs(M[i, i])
        max_row = i
        for k in range(i + 1, n):
            val = math.fabs(M[k, i])
            if val > max_val:
                max_val = val
                max_row = k
        
        if max_row != i:
            for c in range(2 * n):
                temp = M[i, c]
                M[i, c] = M[max_row, c]
                M[max_row, c] = temp
        
        pivot = M[i, i]
        for c in range(2 * n):
            M[i, c] /= pivot
        
        for k in range(n):
            if k != i:
                factor = M[k, i]
                if factor != 0.0:
                    for c in range(2 * n):
                        M[k, c] -= factor * M[i, c]

    inv_A = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            inv_A[i, j] = M[i, n + j]
    return inv_A


def _cholesky(A):
    n = A.shape[0]
    L = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1):
            s = 0.0
            for k in range(j):
                s += L[i, k] * L[j, k]
            
            if i == j:
                val = A[i, i] - s
                if val < 0.0:
                    val = 0.0
                L[i, j] = math.sqrt(val)
            else:
                L[i, j] = (A[i, j] - s) / L[j, j]
    return L


def _transpose(A):
    rows = A.shape[0]
    cols = A.shape[1]
    At = np.zeros((cols, rows), dtype=np.float64)
    for i in range(rows):
        for j in range(cols):
            At[j, i] = A[i, j]
    return At


def _mat_mul(A, B):
    m = A.shape[0]
    k = A.shape[1]
    n = B.shape[1]
    C = np.zeros((m, n), dtype=np.float64)
    for i in range(m):
        for j in range(n):
            s = 0.0
            for p in range(k):
                s += A[i, p] * B[p, j]
            C[i, j] = s
    return C


def _outer(a, b):
    m = len(a)
    n = len(b)
    res = np.zeros((m, n), dtype=np.float64)
    for i in range(m):
        ai = a[i]
        for j in range(n):
            res[i, j] = ai * b[j]
    return res


def _norm(M):
    rows = M.shape[0]
    cols = M.shape[1]
    s = 0.0
    for i in range(rows):
        for j in range(cols):
            val = M[i, j]
            s += val * val
    return math.sqrt(s)


def _row_scales(W, bits):
    qmax = 2 ** (bits - 1) - 1
    rows = W.shape[0]
    cols = W.shape[1]
    scale_list = []
    for i in range(rows):
        max_val = 0.0
        for j in range(cols):
            val = W[i, j]
            abs_val = val if val >= 0.0 else -val
            if abs_val > max_val:
                max_val = abs_val
        s = max_val / qmax
        if s == 0.0:
            s = 1.0
        scale_list.append(s)
    scale = np.array(scale_list, dtype=np.float64)
    return scale, qmax


def _q(w_col, scale, qmax):
    if w_col.ndim == 1:
        res = []
        for i in range(len(w_col)):
            s = scale[i]
            val = w_col[i] / s
            rounded = round(val)
            if rounded < -qmax:
                clipped = -qmax
            elif rounded > qmax:
                clipped = qmax
            else:
                clipped = rounded
            res.append(clipped * s)
        return np.array(res, dtype=np.float64)
    else:
        rows, cols = w_col.shape
        res = []
        for i in range(rows):
            row_res = []
            s = scale[i, 0] if scale.ndim == 2 else scale[i]
            for j in range(cols):
                val = w_col[i, j] / s
                rounded = round(val)
                if rounded < -qmax:
                    clipped = -qmax
                elif rounded > qmax:
                    clipped = qmax
                else:
                    clipped = rounded
                row_res.append(clipped * s)
            res.append(row_res)
        return np.array(res, dtype=np.float64)


def _quantize_rtn(W, bits):
    W = np.asarray(W, dtype=np.float64)
    scale, qmax = _row_scales(W, bits)
    return _q(W, scale[:, None], qmax)


def _quantize_gptq(W, X, bits, damp=0.01):
    W = np.asarray(W, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    d_in = W.shape[1]

    H = _xt_x(X)
    mean_diag = _mean_diag(H)
    eye_mat = _eye(d_in)

    rows_h, cols_h = H.shape
    for i in range(rows_h):
        for j in range(cols_h):
            H[i, j] += damp * mean_diag * eye_mat[i, j]

    scale, qmax = _row_scales(W, bits)
    inv_H = _inv(H)
    L = _cholesky(inv_H)

    n_ch = L.shape[0]
    U = np.zeros((n_ch, n_ch), dtype=np.float64)
    for i in range(n_ch):
        for j in range(n_ch):
            U[i, j] = L[j, i]

    Wc = W.copy()
    Q = np.zeros_like(W)
    for j in range(d_in):
        w = np.array([Wc[r, j] for r in range(Wc.shape[0])], dtype=np.float64)
        q = _q(w, scale, qmax)
        for r in range(Wc.shape[0]):
            Q[r, j] = q[r]
        
        u_jj = U[j, j]
        err = np.array([(w[r] - q[r]) / u_jj for r in range(len(w))], dtype=np.float64)
        
        if j + 1 < d_in:
            sub_cols = d_in - (j + 1)
            outer_res = _outer(err, np.array([U[j, j + 1 + c] for c in range(sub_cols)], dtype=np.float64))
            for r in range(Wc.shape[0]):
                for c_idx in range(sub_cols):
                    Wc[r, j + 1 + c_idx] -= outer_res[r, c_idx]
    return Q


def gptq_vs_rtn_error_ratio(W, X, bits):
    W = np.asarray(W, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)

    W_rtn = _quantize_rtn(W, bits)
    W_gptq = _quantize_gptq(W, X, bits)

    wt_rtn = _transpose(W_rtn)
    wt = _transpose(W)
    wt_gptq = _transpose(W_gptq)

    x_wt_rtn = _mat_mul(X, wt_rtn)
    x_wt = _mat_mul(X, wt)
    x_wt_gptq = _mat_mul(X, wt_gptq)

    rows = x_wt_rtn.shape[0]
    cols = x_wt_rtn.shape[1]

    diff_rtn = np.zeros((rows, cols), dtype=np.float64)
    diff_gptq = np.zeros((rows, cols), dtype=np.float64)

    for i in range(rows):
        for j in range(cols):
            diff_rtn[i, j] = x_wt_rtn[i, j] - x_wt[i, j]
            diff_gptq[i, j] = x_wt_gptq[i, j] - x_wt[i, j]

    err_rtn = _norm(diff_rtn)
    err_gptq = _norm(diff_gptq)

    return float(err_gptq / err_rtn)
