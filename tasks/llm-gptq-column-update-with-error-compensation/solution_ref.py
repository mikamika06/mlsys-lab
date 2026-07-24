import numpy as np

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
    W_mod = W.copy()
    m, n = W.shape
    H = X.T @ X / X.shape[0]  # Hessian approximation
    codes = np.empty_like(W, dtype=np.int8)
    scales = np.zeros(n, dtype=np.float64)

    for j in range(n):
        col = W_mod[:, j]
        scale = np.max(np.abs(col)) / 127.0
        if scale == 0:
            scale = 1.0
        scales[j] = scale

        int_col = np.round(col / scale).astype(np.int8)
        codes[:, j] = int_col

        recon = scale * int_col.astype(np.float64)
        residual = col - recon

        if j + 1 < n:
            factor = H[j, j+1:] / (H[j, j] + 1e-12)  # shape (n-j-1,)
            for k_idx, k in enumerate(range(j + 1, n)):
                W_mod[:, k] += residual * factor[k_idx]

    return codes, scales
