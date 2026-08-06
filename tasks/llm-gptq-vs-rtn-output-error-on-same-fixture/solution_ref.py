import math
import numpy as np


def _rint_half_to_even(x):
    if math.isnan(x) or math.isinf(x):
        return x
    f = math.floor(x)
    r = x - f
    if r < 0.5:
        return float(f)
    elif r > 0.5:
        return float(f + 1)
    else:
        if f % 2 == 0:
            return float(f)
        else:
            return float(f + 1)


def _row_scales(W, bits):
    qmax = 2 ** (bits - 1) - 1
    rows, cols = W.shape
    scale = np.zeros(rows, dtype=np.float64)
    for i in range(rows):
        max_val = 0.0
        for j in range(cols):
            val = abs(float(W[i, j]))
            if val > max_val:
                max_val = val
        s = max_val / qmax
        if s == 0.0:
            s = 1.0
        scale[i] = s
    return scale, qmax


def _q_val(w_val, scale, qmax):
    val = _rint_half_to_even(w_val / scale)
    if val < -qmax:
        val = float(-qmax)
    elif val > qmax:
        val = float(qmax)
    return val * scale


def quantize_rtn(W, bits):
    W = np.asarray(W, dtype=np.float64)
    rows, cols = W.shape
    scale, qmax = _row_scales(W, bits)
    Q = np.zeros((rows, cols), dtype=np.float64)
    for i in range(rows):
        s = scale[i]
        for j in range(cols):
            Q[i, j] = _q_val(W[i, j], s, qmax)
    return Q


def quantize_gptq(W, X, bits, damp=0.01):
    W = np.asarray(W, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    n_samples, d_in = X.shape
    d_out = W.shape[0]

    H = np.zeros((d_in, d_in), dtype=np.float64)
    for i in range(d_in):
        for j in range(d_in):
            s = 0.0
            for k in range(n_samples):
                s += X[k, i] * X[k, j]
            H[i, j] = s

    diag_sum = 0.0
    for i in range(d_in):
        diag_sum += H[i, i]
    diag_mean = diag_sum / float(d_in)

    for i in range(d_in):
        H[i, i] += damp * diag_mean

    scale, qmax = _row_scales(W, bits)

    invH = np.zeros((d_in, d_in), dtype=np.float64)
    for i in range(d_in):
        invH[i, i] = 1.0

    for i in range(d_in):
        pivot = H[i, i]
        for j in range(d_in):
            H[i, j] /= pivot
            invH[i, j] /= pivot
        for k in range(d_in):
            if k != i:
                factor = H[k, i]
                for j in range(d_in):
                    H[k, j] -= factor * H[i, j]
                    invH[k, j] -= factor * invH[i, j]

    L = np.zeros((d_in, d_in), dtype=np.float64)
    for i in range(d_in):
        for j in range(i + 1):
            s = 0.0
            for k in range(j):
                s += L[i, k] * L[j, k]
            if i == j:
                L[i, j] = math.sqrt(invH[i, i] - s)
            else:
                L[i, j] = (invH[i, j] - s) / L[j, j]

    U = np.zeros((d_in, d_in), dtype=np.float64)
    for i in range(d_in):
        for j in range(i, d_in):
            U[i, j] = L[j, i]

    Wc = W.copy()
    Q = np.zeros((d_out, d_in), dtype=np.float64)
    err = np.zeros(d_out, dtype=np.float64)

    for j in range(d_in):
        for i in range(d_out):
            w = Wc[i, j]
            q = _q_val(w, scale[i], qmax)
            Q[i, j] = q
            err[i] = (w - q) / U[j, j]

        if j + 1 < d_in:
            for i in range(d_out):
                e = err[i]
                for k in range(j + 1, d_in):
                    Wc[i, k] -= e * U[j, k]

    return Q
