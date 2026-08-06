import math
import numpy as np


def _affine_quant_dequant(x: np.ndarray, bits: int, axis) -> np.ndarray:
    """Uniform affine (asymmetric) min-max quantizer/dequantizer.

    `axis` selects the grouping: the min/max (and therefore the
    scale/zero-point) are computed per-group along that axis, and every
    element in the group shares that one scale/zero-point. `axis=None`
    means a single group for the whole tensor (per-tensor quant).
    """
    x = np.asarray(x, dtype=np.float64)
    qmax = (1 << bits) - 1
    n, d = x.shape

    if axis is None:
        shape_stat = (1, 1)
    elif axis == 0:
        shape_stat = (1, d)
    elif axis == 1:
        shape_stat = (n, 1)
    else:
        raise ValueError("Invalid axis")

    xmin = np.empty(shape_stat, dtype=np.float64)
    xmax = np.empty(shape_stat, dtype=np.float64)

    if axis is None:
        mn = float("inf")
        mx = float("-inf")
        for i in range(n):
            for j in range(d):
                v = x[i, j]
                if v < mn:
                    mn = v
                if v > mx:
                    mx = v
        xmin[0, 0] = mn
        xmax[0, 0] = mx
    elif axis == 0:
        for j in range(d):
            mn = float("inf")
            mx = float("-inf")
            for i in range(n):
                v = x[i, j]
                if v < mn:
                    mn = v
                if v > mx:
                    mx = v
            xmin[0, j] = mn
            xmax[0, j] = mx
    elif axis == 1:
        for i in range(n):
            mn = float("inf")
            mx = float("-inf")
            for j in range(d):
                v = x[i, j]
                if v < mn:
                    mn = v
                if v > mx:
                    mx = v
            xmin[i, 0] = mn
            xmax[i, 0] = mx

    scale = np.empty(shape_stat, dtype=np.float64)
    if axis is None:
        s = (xmax[0, 0] - xmin[0, 0]) / qmax
        scale[0, 0] = 1.0 if s == 0 else s
    elif axis == 0:
        for j in range(d):
            s = (xmax[0, j] - xmin[0, j]) / qmax
            scale[0, j] = 1.0 if s == 0 else s
    elif axis == 1:
        for i in range(n):
            s = (xmax[i, 0] - xmin[i, 0]) / qmax
            scale[i, 0] = 1.0 if s == 0 else s

    zero_point = np.empty(shape_stat, dtype=np.float64)
    if axis is None:
        zero_point[0, 0] = round(-xmin[0, 0] / scale[0, 0])
    elif axis == 0:
        for j in range(d):
            zero_point[0, j] = round(-xmin[0, j] / scale[0, j])
    elif axis == 1:
        for i in range(n):
            zero_point[i, 0] = round(-xmin[i, 0] / scale[i, 0])

    out = np.empty((n, d), dtype=np.float64)
    for i in range(n):
        for j in range(d):
            if axis is None:
                sc = scale[0, 0]
                zp = zero_point[0, 0]
            elif axis == 0:
                sc = scale[0, j]
                zp = zero_point[0, j]
            else:
                sc = scale[i, 0]
                zp = zero_point[i, 0]

            val = round(x[i, j] / sc + zp)
            if val < 0:
                val = 0.0
            elif val > qmax:
                val = float(qmax)
            else:
                val = float(val)
            out[i, j] = (val - zp) * sc

    return out


def _attention(K: np.ndarray, V: np.ndarray, q: np.ndarray) -> np.ndarray:
    n, d = K.shape
    logits = np.empty(n, dtype=np.float64)
    for i in range(n):
        s = 0.0
        for j in range(d):
            s += K[i, j] * q[j]
        logits[i] = s / math.sqrt(d)

    max_l = float("-inf")
    for i in range(n):
        if logits[i] > max_l:
            max_l = logits[i]

    w = np.empty(n, dtype=np.float64)
    sum_w = 0.0
    for i in range(n):
        val = math.exp(logits[i] - max_l)
        w[i] = val
        sum_w += val

    for i in range(n):
        w[i] /= sum_w

    out = np.empty(d, dtype=np.float64)
    for j in range(d):
        s = 0.0
        for i in range(n):
            s += w[i] * V[i, j]
        out[j] = s

    return out


def kivi_quant_errors(K: np.ndarray, V: np.ndarray, q: np.ndarray, bits: int) -> np.ndarray:
    """
    K, V: (n_tokens, d) fp64 key/value cache. q: (d,) fp64 query.
    bits: quantizer bit-width.

    KIVI quantizes keys PER-CHANNEL (one scale/zero-point per column,
    computed across all tokens) and values PER-TOKEN (one scale/zero-point
    per row, computed across all channels) -- the asymmetric axis choice
    that makes low-bit KV cache quantization viable, because RoPE-rotated
    key channels have consistent per-channel outlier structure while value
    outliers are token-specific.

    Returns np.array([
        k_mse_per_channel,   # MSE of per-channel-quantized K vs true K
        k_mse_per_tensor,    # MSE of a per-tensor-quantized K baseline vs true K
        attn_max_abs_err,    # max abs error of attention(K,V,q) using the
                              # (per-channel K, per-token V) KIVI-quantized
                              # cache, vs the exact fp64 attention output
    ])
    """
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    K_per_channel = _affine_quant_dequant(K, bits, axis=0)
    V_per_token = _affine_quant_dequant(V, bits, axis=1)
    K_per_tensor = _affine_quant_dequant(K, bits, axis=None)

    n, d = K.shape
    total_channel = 0.0
    total_tensor = 0.0
    count = n * d
    for i in range(n):
        for j in range(d):
            diff_c = K_per_channel[i, j] - K[i, j]
            total_channel += diff_c * diff_c
            diff_t = K_per_tensor[i, j] - K[i, j]
            total_tensor += diff_t * diff_t

    k_mse_per_channel = float(total_channel / count)
    k_mse_per_tensor = float(total_tensor / count)

    base = _attention(K, V, q)
    kivi_out = _attention(K_per_channel, V_per_token, q)

    max_err = 0.0
    for j in range(d):
        diff = abs(kivi_out[j] - base[j])
        if diff > max_err:
            max_err = diff
    attn_max_abs_err = float(max_err)

    return np.array([k_mse_per_channel, k_mse_per_tensor, attn_max_abs_err])
