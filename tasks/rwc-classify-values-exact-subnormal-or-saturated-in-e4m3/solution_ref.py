import math

def _decode(bits: int) -> float:
    sign = -1.0 if (bits >> 7) & 1 else 1.0
    exp_bits = (bits >> 3) & 0xF
    mant_bits = bits & 0x7
    bias = 7

    if exp_bits == 0:
        if mant_bits == 0:
            return 0.0 * sign
        # subnormal: value = sign * (mant/8) * 2^-6
        return sign * (mant_bits / 8.0) * (2.0 ** -6)
    elif exp_bits == 15:
        return float('inf') * sign
    else:
        return sign * (1 + mant_bits / 8.0) * (2.0 ** (exp_bits - bias))

def _nearest_e4m3(v: float) -> float:
    best = None
    best_diff = 1e100
    for bits in range(256):
        val = _decode(bits)
        diff = abs(val - v)
        if diff < best_diff:
            best_diff = diff
            best = val
    return best

def classify_e4m3(vals: list[float]) -> list[str]:
    """
    Classify each element of *vals* as 'EXACT', 'SUBNORMAL' or 'SATURATED'
    with respect to the 8‑bit e4m3 floating point format.

    Parameters
    ----------
    vals : list of float
        Input values to classify.

    Returns
    -------
    list of str
        A list of strings of the same length as *vals*.
    """
    out = []
    for v in vals:
        if math.isnan(v) or math.isinf(v) or abs(v) > 448.0:
            out.append('SATURATED')
            continue
        enc = _nearest_e4m3(v)
        if enc == v:
            out.append('EXACT')
        else:
            out.append('SUBNORMAL')
    return out
