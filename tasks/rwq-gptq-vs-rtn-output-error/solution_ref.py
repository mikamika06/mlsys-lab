import numpy as np

BITS_DAMP = 0.01


def _row_grid(W, bits):
    qmax = 2 ** (bits - 1) - 1
    amax = np.max(np.abs(W), axis=1)
    amax = np.where(amax == 0, 1.0, amax)
    scale = amax / qmax
    return scale, qmax


def _rtn(W, bits):
    scale, qmax = _row_grid(W, bits)
    return np.clip(np.round(W / scale[:, None]), -qmax, qmax) * scale[:, None]


def _gptq(W, X, bits, damp=BITS_DAMP):
    n = X.shape[0]
    d_out, d_in = W.shape
    H = (X.T @ X) / n
    H = H + damp * np.mean(np.diag(H)) * np.eye(d_in)
    Hinv = np.linalg.inv(H)
    scale, qmax = _row_grid(W, bits)

    Wq = np.zeros_like(W)
    Werr = W.copy()
    Hinv = Hinv.copy()
    for j in range(d_in):
        d = Hinv[j, j]
        w = Werr[:, j]
        q = np.clip(np.round(w / scale), -qmax, qmax) * scale
        Wq[:, j] = q
        err = (w - q) / d
        if j + 1 < d_in:
            Werr[:, j + 1:] -= np.outer(err, Hinv[j, j + 1:])
            Hinv[j + 1:, j + 1:] -= np.outer(Hinv[j + 1:, j], Hinv[j, j + 1:]) / d
    return Wq


def _mse(X, W, Wq):
    Y = X @ W.T
    Yq = X @ Wq.T
    return float(np.mean((Y - Yq) ** 2))


def gptq_vs_rtn_output_error(W: np.ndarray, X: np.ndarray, bits: int):
    W = np.asarray(W, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)

    Wq_rtn = _rtn(W, bits)
    Wq_gptq = _gptq(W, X, bits)

    mse_rtn = _mse(X, W, Wq_rtn)
    mse_gptq = _mse(X, W, Wq_gptq)
    return mse_rtn, mse_gptq
