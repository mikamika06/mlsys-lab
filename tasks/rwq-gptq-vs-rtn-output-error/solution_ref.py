import numpy as np

BITS_DAMP = 0.01


def _row_grid(W, bits):
    qmax = 2 ** (bits - 1) - 1
    d_out, d_in = W.shape
    scale = np.zeros(d_out, dtype=W.dtype)
    for i in range(d_out):
        amax = 0.0
        for j in range(d_in):
            val = W[i, j]
            if val < 0:
                val = -val
            if val > amax:
                amax = val
        if amax == 0.0:
            amax = 1.0
        scale[i] = amax / qmax
    return scale, qmax


def _rtn(W, bits):
    scale, qmax = _row_grid(W, bits)
    d_out, d_in = W.shape
    Wq = np.zeros_like(W)
    for i in range(d_out):
        s = scale[i]
        for j in range(d_in):
            val = W[i, j] / s
            r = round(val)
            if r < -qmax:
                r = -qmax
            elif r > qmax:
                r = qmax
            Wq[i, j] = r * s
    return Wq


def _invert_matrix(A):
    d = A.shape[0]
    M = np.zeros((d, 2 * d), dtype=A.dtype)
    for i in range(d):
        for j in range(d):
            M[i, j] = A[i, j]
        M[i, d + i] = 1.0

    for i in range(d):
        pivot_row = i
        max_val = abs(M[i, i])
        for r in range(i + 1, d):
            val = abs(M[r, i])
            if val > max_val:
                max_val = val
                pivot_row = r
        if pivot_row != i:
            for c in range(2 * d):
                temp = M[i, c]
                M[i, c] = M[pivot_row, c]
                M[pivot_row, c] = temp
        
        pivot = M[i, i]
        for c in range(2 * d):
            M[i, c] /= pivot
        
        for r in range(d):
            if r != i:
                factor = M[r, i]
                if factor != 0.0:
                    for c in range(2 * d):
                        M[r, c] -= factor * M[i, c]

    inv = np.zeros((d, d), dtype=A.dtype)
    for i in range(d):
        for j in range(d):
            inv[i, j] = M[i, d + j]
    return inv


def _gptq(W, X, bits, damp=BITS_DAMP):
    n = X.shape[0]
    d_out, d_in = W.shape
    
    H = np.zeros((d_in, d_in), dtype=X.dtype)
    for i in range(d_in):
        for k in range(d_in):
            s = 0.0
            for row in range(n):
                s += X[row, i] * X[row, k]
            H[i, k] = s / n

    diag_sum = 0.0
    for i in range(d_in):
        diag_sum += H[i, i]
    mean_diag = diag_sum / d_in
    for i in range(d_in):
        H[i, i] += damp * mean_diag

    Hinv = _invert_matrix(H)
    scale, qmax = _row_grid(W, bits)

    Wq = np.zeros_like(W)
    Werr = W.copy()
    Hinv = Hinv.copy()

    for j in range(d_in):
        d = Hinv[j, j]
        w = np.zeros(d_out, dtype=W.dtype)
        for i in range(d_out):
            w[i] = Werr[i, j]

        q = np.zeros(d_out, dtype=W.dtype)
        for i in range(d_out):
            val = w[i] / scale[i]
            r = round(val)
            if r < -qmax:
                r = -qmax
            elif r > qmax:
                r = qmax
            q[i] = r * scale[i]
            Wq[i, j] = q[i]

        err = np.zeros(d_out, dtype=W.dtype)
        for i in range(d_out):
            err[i] = (w[i] - q[i]) / d

        if j + 1 < d_in:
            cols_remaining = d_in - (j + 1)
            for i in range(d_out):
                for c_idx in range(cols_remaining):
                    col = (j + 1) + c_idx
                    Werr[i, col] -= err[i] * Hinv[j, col]
            
            for r_idx in range(cols_remaining):
                row = (j + 1) + r_idx
                hinv_val = Hinv[row, j]
                for c_idx in range(cols_remaining):
                    col = (j + 1) + c_idx
                    Hinv[row, col] -= (hinv_val * Hinv[j, col]) / d

    return Wq


def _mse(X, W, Wq):
    n = X.shape[0]
    d_in = X.shape[1]
    d_out = W.shape[0]

    Y = np.zeros((n, d_out), dtype=X.dtype)
    for r in range(n):
        for c in range(d_out):
            s = 0.0
            for k in range(d_in):
                s += X[r, k] * W[c, k]
            Y[r, c] = s

    Yq = np.zeros((n, d_out), dtype=X.dtype)
    for r in range(n):
        for c in range(d_out):
            s = 0.0
            for k in range(d_in):
                s += X[r, k] * Wq[c, k]
            Yq[r, c] = s

    total = 0.0
    count = n * d_out
    for r in range(n):
        for c in range(d_out):
            diff = Y[r, c] - Yq[r, c]
            total += diff * diff

    return float(total / count)


def gptq_vs_rtn_output_error(W: np.ndarray, X: np.ndarray, bits: int):
    W = np.asarray(W, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)

    Wq_rtn = _rtn(W, bits)
    Wq_gptq = _gptq(W, X, bits)

    mse_rtn = _mse(X, W, Wq_rtn)
    mse_gptq = _mse(X, W, Wq_gptq)
    return mse_rtn, mse_gptq
