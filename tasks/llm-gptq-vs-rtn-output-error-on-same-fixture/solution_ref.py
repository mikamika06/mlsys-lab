import numpy as np


def _row_scales(W, bits):
    qmax = 2 ** (bits - 1) - 1
    scale = np.max(np.abs(W), axis=1) / qmax
    scale = np.where(scale == 0.0, 1.0, scale)
    return scale, qmax


def _q(w_col, scale, qmax):
    return np.clip(np.rint(w_col / scale), -qmax, qmax) * scale


def quantize_rtn(W, bits):
    W = np.asarray(W, dtype=np.float64)
    scale, qmax = _row_scales(W, bits)
    return _q(W, scale[:, None], qmax)


def quantize_gptq(W, X, bits, damp=0.01):
    W = np.asarray(W, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    d_in = W.shape[1]

    H = X.T @ X
    H = H + damp * float(np.mean(np.diag(H))) * np.eye(d_in)

    # scales frozen on the ORIGINAL weights, per output row
    scale, qmax = _row_scales(W, bits)

    U = np.linalg.cholesky(np.linalg.inv(H)).T  # upper triangular, inv(H) = U^T U

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
