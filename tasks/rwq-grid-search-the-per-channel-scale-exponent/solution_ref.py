import math

RATIOS = tuple(i / 10 for i in range(11))  # 0.0, 0.1, ..., 1.0


def _quantize_symmetric_rows(Wm: list[list[float]], n_bits: int) -> list[list[float]]:
    """Per-output-row symmetric round-to-nearest quantization, dequantized back."""
    qmax = 2 ** (n_bits - 1) - 1
    rows = len(Wm)
    cols = len(Wm[0])
    row_scale = [0.0] * rows
    for i in range(rows):
        max_val = 0.0
        for j in range(cols):
            val = Wm[i][j]
            if val < 0:
                val = -val
            if val > max_val:
                max_val = val
        scale = max_val / qmax
        if scale == 0:
            scale = 1.0
        row_scale[i] = scale

    q = [[0.0] * cols for _ in range(rows)]
    for i in range(rows):
        scale = row_scale[i]
        for j in range(cols):
            val = Wm[i][j] / scale
            rounded = round(val)
            if rounded < -qmax - 1:
                rounded = -qmax - 1
            elif rounded > qmax:
                rounded = qmax
            q[i][j] = float(rounded)

    res = [[0.0] * cols for _ in range(rows)]
    for i in range(rows):
        scale = row_scale[i]
        for j in range(cols):
            res[i][j] = q[i][j] * scale
    return res


def awq_ratio_search(W: list[list[float]], X: list[list[float]], n_bits: int = 4) -> tuple[int, float]:
    """
    Search the fixed AWQ ratio grid RATIOS = (0.0, 0.1, ..., 1.0) for the
    ratio that minimizes the calibration-activation-weighted output MSE
    after quantizing the (scaled) weights to `n_bits`.

    For each ratio r in RATIOS:
      s_x = mean absolute activation per input channel
      s = s_x ** r
      s = s / sqrt(max(s) * min(s))        # keep dynamic range balanced
      W_scaled = W scaled by s per column
      W_hat = dequantize(quantize_per_row(W_scaled, n_bits)) / s per column
      mse = mean squared error between original output and quantized output

    Returns (best_ratio_index, best_mse): the index into RATIOS achieving
    the smallest mse, and that mse value.
    """
    n_samp = len(X)
    in_f = len(X[0])
    out_f = len(W)

    Y = [[0.0] * out_f for _ in range(n_samp)]
    for i in range(n_samp):
        for j in range(out_f):
            acc = 0.0
            for k in range(in_f):
                acc += X[i][k] * W[j][k]
            Y[i][j] = acc

    s_x = [0.0] * in_f
    for k in range(in_f):
        acc = 0.0
        for i in range(n_samp):
            val = X[i][k]
            if val < 0:
                val = -val
            acc += val
        s_x[k] = acc / n_samp

    for k in range(in_f):
        if s_x[k] == 0:
            s_x[k] = 1e-12

    mses = []
    for r_idx, r in enumerate(RATIOS):
        s = [0.0] * in_f
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

        Wsc = [[0.0] * in_f for _ in range(out_f)]
        for i in range(out_f):
            for k in range(in_f):
                Wsc[i][k] = W[i][k] * s[k]

        Wq = _quantize_symmetric_rows(Wsc, n_bits)

        What = [[0.0] * in_f for _ in range(out_f)]
        for i in range(out_f):
            for k in range(in_f):
                What[i][k] = Wq[i][k] / s[k]

        Yhat = [[0.0] * out_f for _ in range(n_samp)]
        for i in range(n_samp):
            for j in range(out_f):
                acc = 0.0
                for k in range(in_f):
                    acc += X[i][k] * What[j][k]
                Yhat[i][j] = acc

        mse_acc = 0.0
        count = 0
        for i in range(n_samp):
            for j in range(out_f):
                diff = Y[i][j] - Yhat[i][j]
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
