import math
import numpy as np

RATIOS = tuple(i / 10 for i in range(11))  # 0.0, 0.1, ..., 1.0


def _quantize_symmetric_rows(Wm: np.ndarray, n_bits: int) -> np.ndarray:
    """Per-output-row symmetric round-to-nearest quantization, dequantized back."""
    qmax = 2 ** (n_bits - 1) - 1
    rows, cols = Wm.shape
    row_scale = np.zeros((rows, 1), dtype=np.float64)
    for i in range(rows):
        max_val = 0.0
        for j in range(cols):
            val = Wm[i, j]
            if val < 0:
                val = -val
            if val > max_val:
                max_val = val
        scale = max_val / qmax
        if scale == 0:
            scale = 1.0
        row_scale[i, 0] = scale

    q = np.zeros((rows, cols), dtype=np.float64)
    for i in range(rows):
        scale = row_scale[i, 0]
        for j in range(cols):
            val = Wm[i, j] / scale
            rounded = round(val)
            if rounded < -qmax - 1:
                rounded = -qmax - 1
            elif rounded > qmax:
                rounded = qmax
            q[i, j] = rounded

    res = np.zeros((rows, cols), dtype=np.float64)
    for i in range(rows):
        scale = row_scale[i, 0]
        for j in range(cols):
            res[i, j] = q[i, j] * scale
    return res


def awq_ratio_search(W: np.ndarray, X: np.ndarray, n_bits: int = 4):
    """
    Search the fixed AWQ ratio grid RATIOS = (0.0, 0.1, ..., 1.0) for the
    ratio that minimizes the calibration-activation-weighted output MSE
    after quantizing the (scaled) weights to `n_bits`.

    For each ratio r in RATIOS:
      s_x = mean(|X|, axis=0)              # per-input-channel activation scale
      s = s_x ** r
      s = s / sqrt(s.max() * s.min())      # keep dynamic range balanced
      W_scaled = W * s[None, :]
      W_hat = dequantize(quantize_per_row(W_scaled, n_bits)) / s[None, :]
      mse = mean((X @ W.T - X @ W_hat.T) ** 2)

    Returns (best_ratio_index, best_mse): the index into RATIOS achieving
    the smallest mse, and that mse value.
    """
    W64 = np.asarray(W, dtype=np.float64)
    X64 = np.asarray(X, dtype=np.float64)
    
    n_samp, in_f = X64.shape
    out_f = W64.shape[0]

    Y = np.zeros((n_samp, out_f), dtype=np.float64)
    for i in range(n_samp):
        for j in range(out_f):
            acc = 0.0
            for k in range(in_f):
                acc += X64[i, k] * W64[j, k]
            Y[i, j] = acc

    s_x = np.zeros((in_f,), dtype=np.float64)
    for k in range(in_f):
        acc = 0.0
        for i in range(n_samp):
            val = X64[i, k]
            if val < 0:
                val = -val
            acc += val
        s_x[k] = acc / n_samp

    for k in range(in_f):
        if s_x[k] == 0:
            s_x[k] = 1e-12

    mses = []
    for r_idx, r in enumerate(RATIOS):
        s = np.zeros((in_f,), dtype=np.float64)
        for k in range(in_f):
            s[k] = s_x[k] ** r

        max_s = s[0]
        min_s = s[0]
        for k in range(1, in_f):
            val = s[k]
            if val > max_s:
                max_s = val
            if val < min_s:
                min_s = val

        denom = math.sqrt(max_s * min_s)
        for k in range(in_f):
            s[k] = s[k] / denom

        Wsc = np.zeros((out_f, in_f), dtype=np.float64)
        for i in range(out_f):
            for k in range(in_f):
                Wsc[i, k] = W64[i, k] * s[k]

        Wq = _quantize_symmetric_rows(Wsc, n_bits)

        What = np.zeros((out_f, in_f), dtype=np.float64)
        for i in range(out_f):
            for k in range(in_f):
                What[i, k] = Wq[i, k] / s[k]

        Yhat = np.zeros((n_samp, out_f), dtype=np.float64)
        for i in range(n_samp):
            for j in range(out_f):
                acc = 0.0
                for k in range(in_f):
                    acc += X64[i, k] * What[j, k]
                Yhat[i, j] = acc

        mse_acc = 0.0
        count = 0
        for i in range(n_samp):
            for j in range(out_f):
                diff = Y[i, j] - Yhat[i, j]
                mse_acc += diff * diff
                count += 1
        mses.append(mse_acc / count)

    best_idx = 0
    best_mse = mses[0]
    for idx in range(1, len(mses)):
        if mses[idx] < best_mse:
            best_mse = mses[idx]
            best_idx = idx

    return best_idx, float(best_mse)
