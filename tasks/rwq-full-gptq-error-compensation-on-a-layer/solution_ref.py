import math
import numpy as np


def _col_scale_zp(w, nbits):
    qmax = (1 << nbits) - 1
    mn = 0.0
    for val in w:
        if val < mn:
            mn = val
    mn = min(0.0, float(mn))
    mx = 0.0
    for val in w:
        if val > mx:
            mx = val
    mx = max(0.0, float(mx))
    scale = (mx - mn) / qmax if mx > mn else 1.0
    val_zp = round(-mn / scale)
    if val_zp < 0:
        val_zp = 0
    elif val_zp > qmax:
        val_zp = qmax
    zp = int(val_zp)
    return scale, zp


def _quant_val(w, scale, zp, nbits):
    qmax = (1 << nbits) - 1
    out = []
    for val in w:
        code = round(val / scale) + zp
        if code < 0:
            code = 0
        elif code > qmax:
            code = qmax
        out.append((code - zp) * scale)
    return np.array(out, dtype=np.float64)


def _invert_matrix(A):
    n = A.shape[0]
    M = []
    for i in range(n):
        row = list(A[i]) + [1.0 if i == j else 0.0 for j in range(n)]
        M.append(row)
    for i in range(n):
        max_el = abs(M[i][i])
        max_row = i
        for k in range(i + 1, n):
            if abs(M[k][i]) > max_el:
                max_el = abs(M[k][i])
                max_row = k
        M[i], M[max_row] = M[max_row], M[i]
        pivot = M[i][i]
        for j in range(2 * n):
            M[i][j] /= pivot
        for k in range(n):
            if k != i:
                factor = M[k][i]
                for j in range(2 * n):
                    M[k][j] -= factor * M[i][j]
    inv_A = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            inv_A[i, j] = M[i][j + n]
    return inv_A


def gptq_quantize_layer(W: np.ndarray, X: np.ndarray, nbits: int, damp: float):
    """
    Full GPTQ, natural (left-to-right) column order:

    1. H = X^T X (calibration Hessian).
    2. Dampen its diagonal by damp * mean(diag(H)), invert -> Hinv.
    3. Per-column affine quant params (scale/zero-point) computed once from
       the ORIGINAL W.
    4. For i = 0 .. d_in-1: quantize column i, compute the error scaled by
       1/Hinv[i,i], subtract its outer-product correction from all
       not-yet-quantized columns (i+1:).
    5. Return (Wq, mse) where mse is the mean squared layer-output error
       mean((X @ Wq^T - X @ W^T)^2).
    """
    W = np.asarray(W, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    W_orig = W.copy()
    d_out, d_in = W.shape

    m_x, n_x = X.shape
    H = np.zeros((d_in, d_in), dtype=np.float64)
    for i in range(d_in):
        for j in range(d_in):
            s = 0.0
            for l in range(m_x):
                s += X[l, i] * X[l, j]
            H[i, j] = s

    Hp = H.copy()
    diag_sum = 0.0
    for j in range(d_in):
        diag_sum += Hp[j, j]
    damp_val = damp * (diag_sum / d_in)
    for j in range(d_in):
        Hp[j, j] += damp_val

    Hinv = _invert_matrix(Hp)

    scale_zp = [_col_scale_zp(W[:, c], nbits) for c in range(d_in)]

    Wq = W.copy()
    for i in range(d_in):
        w_col = Wq[:, i]
        scale, zp = scale_zp[i]
        q_col = _quant_val(w_col, scale, zp, nbits)
        err = (w_col - q_col) / Hinv[i, i]
        Wq[:, i] = q_col
        if i + 1 < d_in:
            for r in range(d_out):
                for c in range(i + 1, d_in):
                    Wq[r, c] -= err[r] * Hinv[i, c]

    Yh = np.zeros((m_x, d_out), dtype=np.float64)
    for i in range(m_x):
        for j in range(d_out):
            s = 0.0
            for l in range(d_in):
                s += X[i, l] * Wq[j, l]
            Yh[i, j] = s

    Y = np.zeros((m_x, d_out), dtype=np.float64)
    for i in range(m_x):
        for j in range(d_out):
            s = 0.0
            for l in range(d_in):
                s += X[i, l] * W_orig[j, l]
            Y[i, j] = s

    mse_sum = 0.0
    total_elements = m_x * d_out
    for i in range(m_x):
        for j in range(d_out):
            diff = Yh[i, j] - Y[i, j]
            mse_sum += diff * diff
    mse = mse_sum / total_elements

    return Wq, mse
