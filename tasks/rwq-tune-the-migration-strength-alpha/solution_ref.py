import math
import numpy as np

EPS = 1e-8


def _w8a8_mse(W, X, alpha):
    n_cal, d_in = X.shape
    d_out, _ = W.shape

    amax_x = np.zeros(d_in, dtype=np.float64)
    for j in range(d_in):
        m = 0.0
        for i in range(n_cal):
            val = abs(X[i, j])
            if val > m:
                m = val
        amax_x[j] = m

    amax_w = np.zeros(d_in, dtype=np.float64)
    for j in range(d_in):
        m = 0.0
        for i in range(d_out):
            val = abs(W[i, j])
            if val > m:
                m = val
        amax_w[j] = m

    s = np.zeros(d_in, dtype=np.float64)
    for j in range(d_in):
        num = math.pow(amax_x[j], alpha)
        den = math.pow(amax_w[j], 1.0 - alpha) + EPS
        val = num / den
        if val < EPS:
            val = EPS
        s[j] = val

    Xs = np.zeros((n_cal, d_in), dtype=np.float64)
    for i in range(n_cal):
        for j in range(d_in):
            Xs[i, j] = X[i, j] / s[j]

    Ws = np.zeros((d_out, d_in), dtype=np.float64)
    for i in range(d_out):
        for j in range(d_in):
            Ws[i, j] = W[i, j] * s[j]

    sx = 0.0
    for i in range(n_cal):
        for j in range(d_in):
            val = abs(Xs[i, j])
            if val > sx:
                sx = val
    sx = sx / 127.0 if sx > 0 else 1.0

    Xq = np.zeros((n_cal, d_in), dtype=np.float64)
    for i in range(n_cal):
        for j in range(d_in):
            q = round(Xs[i, j] / sx)
            if q < -127:
                q = -127
            elif q > 127:
                q = 127
            Xq[i, j] = q * sx

    aw = np.zeros(d_out, dtype=np.float64)
    for i in range(d_out):
        m = 0.0
        for j in range(d_in):
            val = abs(Ws[i, j])
            if val > m:
                m = val
        aw[i] = m

    sw = np.zeros(d_out, dtype=np.float64)
    for i in range(d_out):
        sw[i] = aw[i] / 127.0 if aw[i] > 0 else 1.0

    Wq = np.zeros((d_out, d_in), dtype=np.float64)
    for i in range(d_out):
        for j in range(d_in):
            q = round(Ws[i, j] / sw[i])
            if q < -127:
                q = -127
            elif q > 127:
                q = 127
            Wq[i, j] = q * sw[i]

    Yhat = np.zeros((n_cal, d_out), dtype=np.float64)
    for i in range(n_cal):
        for r in range(d_out):
            acc = 0.0
            for j in range(d_in):
                acc += Xq[i, j] * Wq[r, j]
            Yhat[i, r] = acc

    Y = np.zeros((n_cal, d_out), dtype=np.float64)
    for i in range(n_cal):
        for r in range(d_out):
            acc = 0.0
            for j in range(d_in):
                acc += X[i, j] * W[r, j]
            Y[i, r] = acc

    total_diff_sq = 0.0
    count = 0
    for i in range(n_cal):
        for r in range(d_out):
            diff = Yhat[i, r] - Y[i, r]
            total_diff_sq += diff * diff
            count += 1

    return float(total_diff_sq / count)


def sweep_alpha(W: np.ndarray, X: np.ndarray, alphas: np.ndarray):
    """
    For each alpha in `alphas`:
      1. Per-input-channel smoothing scale
         s_j = max(|X[:,j]|)^alpha / max(|W[:,j]|)^(1-alpha).
      2. Smooth: X' = X / s, W' = W * s.
      3. Quantize X' to int8 with one dynamic per-tensor scale
         (max(|X'|)/127); quantize W' to int8 with one scale per output
         row (max(|W'[row]|)/127). Dequantize both.
      4. Output MSE = mean((X'_deq @ W'_deq^T - X @ W^T)^2).

    Returns (best_idx, best_mse): the index into `alphas` achieving the
    lowest output MSE, and that MSE.
    """
    W = np.asarray(W, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    
    mses = []
    for a in alphas:
        mses.append(_w8a8_mse(W, X, float(a)))
        
    best_idx = 0
    best_mse = mses[0]
    for idx in range(1, len(mses)):
        if mses[idx] < best_mse:
            best_mse = mses[idx]
            best_idx = idx
            
    return int(best_idx), float(best_mse)
