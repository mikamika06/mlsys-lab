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
    return np.unique(np.array(vals, dtype=np.float64))


def _fp8_quant(x, scale, exp_bits, mant_bits, bias):
    levels = _fp_values(exp_bits, mant_bits, bias)
    y = np.abs(x / scale)
    idx = np.searchsorted(levels, y)
    idx = np.clip(idx, 1, len(levels) - 1)
    left = levels[idx - 1]
    right = levels[idx]
    q = np.where(y - left <= right - y, left, right)
    return np.sign(x) * q * scale


def _int8_quant(x, scale):
    return np.clip(np.rint(x / scale), -127, 127) * scale


def _error(x, y):
    return float(np.linalg.norm(x - y) / (np.linalg.norm(x) + 1e-12))


def _best_scale(x, quantize, max_value):
    base = np.max(np.abs(x)) / max_value
    best = None
    for multiplier in np.geomspace(0.25, 4.0, 41):
        scale = base * multiplier
        err = _error(x, quantize(scale))
        if best is None or err < best:
            best = err
    return best


def compare_quant_formats(weights):
    x = np.asarray(weights, dtype=np.float64)

    e4 = _best_scale(
        x,
        lambda s: _fp8_quant(x, s, 4, 3, 7),
        np.max(_fp_values(4, 3, 7)),
    )
    e5 = _best_scale(
        x,
        lambda s: _fp8_quant(x, s, 5, 2, 15),
        np.max(_fp_values(5, 2, 15)),
    )
    i8 = _best_scale(x, lambda s: _int8_quant(x, s), 127.0)

    errors = {"e4m3": e4, "e5m2": e5, "int8": i8}
    return {
        "e4m3_error": e4,
        "e5m2_error": e5,
        "int8_error": i8,
        "best_format": min(errors, key=errors.get),
    }
