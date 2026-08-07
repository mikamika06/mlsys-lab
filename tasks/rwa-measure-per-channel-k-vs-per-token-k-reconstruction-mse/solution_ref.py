def _affine_quant_dequant(x: list[list[float]], bits: int, axis: int) -> list[list[float]]:
    """Uniform affine (asymmetric) min-max quantizer/dequantizer.

    The min/max (and therefore the scale/zero-point) are computed
    per-group along `axis`; every element in a group shares that one
    scale/zero-point.
    """
    qmax = (1 << bits) - 1
    rows = len(x)
    cols = len(x[0])
    out = [[0.0] * cols for _ in range(rows)]

    if axis == 0:
        xmin = [0.0] * cols
        xmax = [0.0] * cols
        for c in range(cols):
            mn = x[0][c]
            mx = x[0][c]
            for r in range(1, rows):
                val = x[r][c]
                if val < mn:
                    mn = val
                if val > mx:
                    mx = val
            xmin[c] = mn
            xmax[c] = mx

        scale = [0.0] * cols
        for c in range(cols):
            s = (xmax[c] - xmin[c]) / qmax
            if s == 0.0:
                s = 1.0
            scale[c] = s

        zero_point = [0.0] * cols
        for c in range(cols):
            zero_point[c] = round(-xmin[c] / scale[c])

        for c in range(cols):
            s = scale[c]
            zp = zero_point[c]
            for r in range(rows):
                q = round(x[r][c] / s + zp)
                if q < 0:
                    q = 0.0
                elif q > qmax:
                    q = float(qmax)
                else:
                    q = float(q)
                out[r][c] = (q - zp) * s

    elif axis == 1:
        xmin = [0.0] * rows
        xmax = [0.0] * rows
        for r in range(rows):
            mn = x[r][0]
            mx = x[r][0]
            for c in range(1, cols):
                val = x[r][c]
                if val < mn:
                    mn = val
                if val > mx:
                    mx = val
            xmin[r] = mn
            xmax[r] = mx

        scale = [0.0] * rows
        for r in range(rows):
            s = (xmax[r] - xmin[r]) / qmax
            if s == 0.0:
                s = 1.0
            scale[r] = s

        zero_point = [0.0] * rows
        for r in range(rows):
            zero_point[r] = round(-xmin[r] / scale[r])

        for r in range(rows):
            s = scale[r]
            zp = zero_point[r]
            for c in range(cols):
                q = round(x[r][c] / s + zp)
                if q < 0:
                    q = 0.0
                elif q > qmax:
                    q = float(qmax)
                else:
                    q = float(q)
                out[r][c] = (q - zp) * s

    return out


def per_channel_vs_per_token_k_mse(K: list[list[float]], bits: int) -> list[float]:
    """
    K: (n_tokens, d_channels) fp64 key cache as list of lists. bits: quantizer bit-width.

    Quantize K two ways with a uniform affine min-max quantizer:
      - per-channel: one scale/zero-point per COLUMN, min/max taken across
        all tokens (axis=0).
      - per-token: one scale/zero-point per ROW, min/max taken across all
        channels (axis=1).

    Returns [mse_per_channel, mse_per_token], the reconstruction
    MSE (mean squared error of dequant(quant(K)) vs K) of each scheme.
    """
    K_per_channel = _affine_quant_dequant(K, bits, axis=0)
    K_per_token = _affine_quant_dequant(K, bits, axis=1)

    rows = len(K)
    cols = len(K[0])
    total_elements = rows * cols

    sum_sq_channel = 0.0
    for r in range(rows):
        for c in range(cols):
            diff = K_per_channel[r][c] - K[r][c]
            sum_sq_channel += diff * diff
    mse_per_channel = float(sum_sq_channel / total_elements)

    sum_sq_token = 0.0
    for r in range(rows):
        for c in range(cols):
            diff = K_per_token[r][c] - K[r][c]
            sum_sq_token += diff * diff
    mse_per_token = float(sum_sq_token / total_elements)

    return [mse_per_channel, mse_per_token]
