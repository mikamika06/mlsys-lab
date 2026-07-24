import numpy as np

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
        return np.inf * sign
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

def ref_classify(vals: np.ndarray) -> np.ndarray:
    vals = np.asarray(vals, dtype=np.float64)
    out = []
    for v in vals:
        if np.isnan(v) or np.isinf(v) or abs(v) > 448.0:
            out.append('SATURATED')
            continue
        enc = _nearest_e4m3(v)
        if enc == v:
            out.append('EXACT')
        else:
            out.append('SUBNORMAL')
    return np.array(out, dtype='<U10')

def grade(sol, fx) -> dict:
    # Test cases covering all categories
    cases = [
        (np.array([0.0]), np.array(['EXACT'])),
        (np.array([0.01]), np.array(['SUBNORMAL'])),
        (np.array([1.0]), np.array(['EXACT'])),
        (np.array([1000.0]), np.array(['SATURATED'])),
        (np.array([float('nan')]), np.array(['SATURATED'])),
        (np.array([-0.02, 2.5, 500.0]),
         np.array(['SUBNORMAL', 'EXACT', 'SATURATED']))
    ]

    ok = 1.0
    for vals, expected in cases:
        try:
            got = sol.classify_e4m3(vals)
            if not isinstance(got, np.ndarray):
                got = np.asarray(got)
            if not np.array_equal(got, expected):
                ok = 0.0
                break
        except Exception:
            ok = 0.0
            break

    return {"exact_match": ok}
