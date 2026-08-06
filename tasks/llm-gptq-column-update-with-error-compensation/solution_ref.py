import numpy as np
import math

def gptq_quantize(W: np.ndarray, X: np.ndarray):
    """
    Quantize each column of W using GPTQ with error compensation.

    Parameters
    ----------
    W : np.ndarray
        Weight matrix of shape (m, n), dtype float32.
    X : np.ndarray
        Activation matrix of shape (b, n), dtype float32.

    Returns
    -------
    codes : np.ndarray
        Integer codes of shape (m, n), dtype int8 in [-127, 127].
    scales : np.ndarray
        Scale factors of shape (n,), dtype float64.
    """
    m, n = W.shape
    b = X.shape[0]

    W_mod = np.empty((m, n), dtype=W.dtype)
    for i in range(m):
        for j in range(n):
            W_mod[i, j] = W[i, j]

    H = np.empty((n, n), dtype=X.dtype)
    for i in range(n):
        for j in range(n):
            acc = 0.0
            for k in range(b):
                acc += X[k, i] * X[k, j]
            H[i, j] = acc / b

    codes = np.empty((m, n), dtype=np.int8)
    scales = np.zeros(n, dtype=np.float64)

    for j in range(n):
        max_abs = 0.0
        for i in range(m):
            val = abs(float(W_mod[i, j]))
            if val > max_abs:
                max_abs = val
        
        scale = max_abs / 127.0
        if scale == 0.0:
            scale = 1.0
        scales[j] = scale

        factors = []
        if j + 1 < n:
            denom = float(H[j, j]) + 1e-12
            for k in range(j + 1, n):
                factors.append(float(H[j, k]) / denom)

        for i in range(m):
            col_val = float(W_mod[i, j])
            int_col = int(round(col_val / scale))
            codes[i, j] = int_col
            
            recon = scale * float(int_col)
            residual = col_val - recon

            if j + 1 < n:
                for k_idx, k in enumerate(range(j + 1, n)):
                    W_mod[i, k] += residual * factors[k_idx]

    return codes, scales
