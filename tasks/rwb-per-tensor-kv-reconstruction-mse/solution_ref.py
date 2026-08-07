import math


def _flatten(x):
    """Recursively flatten a nested list of arbitrary depth."""
    if isinstance(x, (list, tuple)):
        flat = []
        for item in x:
            flat.extend(_flatten(item))
        return flat
    else:
        return [float(x)]


def _reconstruct_shape(flat_vals, original):
    """Reconstruct the nested structure of original using flat_vals."""
    if not isinstance(original, (list, tuple)):
        return flat_vals.pop(0)
    res = []
    for item in original:
        res.append(_reconstruct_shape(flat_vals, item))
    return res


def _e4m3_roundtrip(x, scale: float):
    """Per-tensor absmax-scaled E4M3 (4 exponent bits, 3 mantissa bits,
    max representable magnitude 448) quantize-then-dequantize."""
    flat_x = _flatten(x)
    flat_out = []

    for val_x in flat_x:
        y = val_x / scale
        if y > 448.0:
            y = 448.0
        elif y < -448.0:
            y = -448.0

        if y > 0.0:
            sign = 1.0
        elif y < 0.0:
            sign = -1.0
        else:
            sign = 0.0

        ay = abs(y)

        max_ay = ay if ay > 0.001953125 else 0.001953125
        exp = math.floor(math.log2(max_ay))
        if exp < -6:
            exp = -6
        elif exp > 7:
            exp = 7

        pow2_exp = math.pow(2.0, exp)
        frac = ay / pow2_exp - 1.0
        mant = round(frac * 8.0) / 8.0
        val = (1.0 + mant) * pow2_exp

        if ay < 0.015625:
            val = round(ay / 0.001953125) * 0.001953125

        if ay == 0.0:
            val = 0.0

        flat_out.append(sign * val * scale)

    return _reconstruct_shape(flat_out, x)


def kv_fp8_reconstruction_mse(K: list[list[float]], V: list[list[float]]) -> dict:
    """Quantize K and V to E4M3 with an independent PER-TENSOR absmax
    scale for each (scale = max(|X|) / 448), dequantize, and report each
    tensor's reconstruction MSE.

    K, V : arbitrary-shape float arrays (nested lists).

    Returns {"mse_k": float, "mse_v": float}.
    """
    flat_k = _flatten(K)
    max_k = 0.0
    for val in flat_k:
        v_abs = abs(float(val))
        if v_abs > max_k:
            max_k = v_abs

    sk = max_k / 448.0
    if sk < 1e-12:
        sk = 1e-12

    flat_v = _flatten(V)
    max_v = 0.0
    for val in flat_v:
        v_abs = abs(float(val))
        if v_abs > max_v:
            max_v = v_abs

    sv = max_v / 448.0
    if sv < 1e-12:
        sv = 1e-12

    K_hat = _e4m3_roundtrip(K, sk)
    V_hat = _e4m3_roundtrip(V, sv)

    flat_k_hat = _flatten(K_hat)
    sum_sq_k = 0.0
    for i in range(len(flat_k)):
        diff = float(flat_k_hat[i]) - float(flat_k[i])
        sum_sq_k += diff * diff
    mse_k = float(sum_sq_k / len(flat_k))

    flat_v_hat = _flatten(V_hat)
    sum_sq_v = 0.0
    for i in range(len(flat_v)):
        diff = float(flat_v_hat[i]) - float(flat_v[i])
        sum_sq_v += diff * diff
    mse_v = float(sum_sq_v / len(flat_v))

    return {"mse_k": mse_k, "mse_v": mse_v}
