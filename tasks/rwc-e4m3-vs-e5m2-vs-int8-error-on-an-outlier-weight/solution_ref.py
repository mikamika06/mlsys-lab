import math
import numpy as np


def _fp_values(exp_bits, mant_bits, bias):
    vals = []
    for e in range(1 << exp_bits):
        for m in range(1 << mant_bits):
            if e == 0:
                if m == 0:
                    v = 0.0
                else:
                    v = m * (2.0 ** (1 - bias - mant_bits))
            else:
                v = (1.0 + m / (2 ** mant_bits)) * (2.0 ** (e - bias))
            vals.append(v)
    unique_vals = sorted(list(set(vals)))
    return np.array(unique_vals, dtype=np.float64)


def _fp8_quant(x, scale, exp_bits, mant_bits, bias):
    levels = _fp_values(exp_bits, mant_bits, bias)
    out = np.empty_like(x, dtype=np.float64)
    for i, val in enumerate(x.flat):
        y_val = abs(val / scale)
        idx = 0
        while idx < len(levels) and levels[idx] < y_val:
            idx += 1
        if idx < 1:
            idx = 1
        elif idx >= len(levels):
            idx = len(levels) - 1
        left = levels[idx - 1]
        right = levels[idx]
        if (y_val - left) <= (right - y_val):
            q = left
        else:
            q = right
        if val < 0:
            sign_val = -1.0
        elif val > 0:
            sign_val = 1.0
        else:
            sign_val = 0.0
        out.flat[i] = sign_val * q * scale
    return out


def _int8_quant(x, scale):
    out = np.empty_like(x, dtype=np.float64)
    for i, val in enumerate(x.flat):
        div = val / scale
        r = round(div)
        if r < -127:
            r = -127
        elif r > 127:
            r = 127
        out.flat[i] = r * scale
    return out


def _error(x, y):
    sum_sq_diff = 0.0
    sum_sq_x = 0.0
    for vx, vy in zip(x.flat, y.flat):
        diff = vx - vy
        sum_sq_diff += diff * diff
        sum_sq_x += vx * vx
    norm_diff = math.sqrt(sum_sq_diff)
    norm_x = math.sqrt(sum_sq_x)
    return float(norm_diff / (norm_x + 1e-12))


def _best_scale(x, quantize, max_value):
    max_abs = 0.0
    for val in x.flat:
        a = abs(val)
        if a > max_abs:
            max_abs = a
    base = max_abs / max_value
    best = None
    for multiplier in [0.25 * math.pow(16.0, i / 40.0) for i in range(41)]:
        scale = base * multiplier
        err = _error(x, quantize(scale))
        if best is None or err < best:
            best = err
    return best


def compare_quant_formats(weights):
    x = np.asarray(weights, dtype=np.float64)

    e4_vals = _fp_values(4, 3, 7)
    e5_vals = _fp_values(5, 2, 15)

    e4_max = -float("inf")
    for v in e4_vals.flat:
        if v > e4_max:
            e4_max = v

    e5_max = -float("inf")
    for v in e5_vals.flat:
        if v > e5_max:
            e5_max = v

    e4 = _best_scale(
        x,
        lambda s: _fp8_quant(x, s, 4, 3, 7),
        e4_max,
    )
    e5 = _best_scale(
        x,
        lambda s: _fp8_quant(x, s, 5, 2, 15),
        e5_max,
    )
    i8 = _best_scale(x, lambda s: _int8_quant(x, s), 127.0)

    errors = {"e4m3": e4, "e5m2": e5, "int8": i8}
    return {
        "e4m3_error": e4,
        "e5m2_error": e5,
        "int8_error": i8,
        "best_format": min(errors, key=errors.get),
    }
