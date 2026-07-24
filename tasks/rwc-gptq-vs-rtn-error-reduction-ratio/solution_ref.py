import numpy as np


def _row_scales(W, bits):
    qmax = 2 ** (bits - 1) - 1
    scale = np.max(np.abs(W), axis=1) / qmax
    scale = np.where(scale == 0.0, 1.0, scale)
    return scale, qmax


def _q(w_col, scale, qmax):
    return np.clip(np.rint(w_col / scale), -qmax, qmax) * scale


def _quantize_rtn(W, bits):
    W = np.asarray(W, dtype=np.float64)
    scale, qmax = _row_scales(W, bits)
    return _q(W, scale[:, None], qmax)


def _quantize_gptq(W, X, bits, damp=0.01):
    W = np.asarray(W, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    d_in = W.shape[1]

    H = X.T @ X
    H = H + damp * float(np.mean(np.diag(H))) * np.eye(d_in)

    scale, qmax = _row_scales(W, bits)
    U = np.linalg.cholesky(np.linalg.inv(H)).T

    Wc = W.copy()
    Q = np.zeros_like(W)
    for j in range(d_in):
        w = Wc[:, j]
        q = _q(w, scale, qmax)
        Q[:, j] = q
        err = (w - q) / U[j, j]
        if j + 1 < d_in:
            Wc[:, j + 1:] -= np.outer(err, U[j, j + 1:])
    return Q


def gptq_vs_rtn_error_ratio(W, X, bits):
    W = np.asarray(W, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)

    W_rtn = _quantize_rtn(W, bits)
    W_gptq = _quantize_gptq(W, X, bits)

    err_rtn = np.linalg.norm(X @ W_rtn.T - X @ W.T)
    err_gptq = np.linalg.norm(X @ W_gptq.T - X @ W.T)

    return float(err_gptq / err_rtn)
