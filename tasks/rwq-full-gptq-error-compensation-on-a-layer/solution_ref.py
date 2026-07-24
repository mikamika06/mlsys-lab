import numpy as np


def _col_scale_zp(w, nbits):
    qmax = (1 << nbits) - 1
    mn = min(0.0, float(np.min(w)))
    mx = max(0.0, float(np.max(w)))
    scale = (mx - mn) / qmax if mx > mn else 1.0
    zp = int(np.clip(round(-mn / scale), 0, qmax))
    return scale, zp


def _quant_val(w, scale, zp, nbits):
    qmax = (1 << nbits) - 1
    codes = np.clip(np.round(w / scale) + zp, 0, qmax)
    return (codes - zp) * scale


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

    H = X.T @ X
    Hp = H.copy()
    damp_val = damp * float(np.mean(np.diag(Hp)))
    Hp[np.diag_indices(d_in)] += damp_val
    Hinv = np.linalg.inv(Hp)

    scale_zp = [_col_scale_zp(W[:, c], nbits) for c in range(d_in)]

    Wq = W.copy()
    for i in range(d_in):
        w_col = Wq[:, i]
        scale, zp = scale_zp[i]
        q_col = _quant_val(w_col, scale, zp, nbits)
        err = (w_col - q_col) / Hinv[i, i]
        Wq[:, i] = q_col
        if i + 1 < d_in:
            Wq[:, i + 1:] -= np.outer(err, Hinv[i, i + 1:])

    Yh = X @ Wq.T
    Y = X @ W_orig.T
    mse = float(np.mean((Yh - Y) ** 2))
    return Wq, mse
