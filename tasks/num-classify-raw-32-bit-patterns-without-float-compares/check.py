import numpy as np

def _oracle(arr):
    sign = (arr >> 31) & 1
    exp = (arr >> 23) & 0xFF
    frac = arr & 0x7FFFFF

    nan_mask = (exp == 255) & (frac != 0)
    inf_mask = (exp == 255) & (frac == 0)
    zero_mask = (exp == 0) & (frac == 0)
    subnormal_mask = (exp == 0) & (frac != 0)
    normal_mask = (~nan_mask & ~inf_mask & ~zero_mask & ~subnormal_mask)

    cat = np.empty_like(arr, dtype=np.int8)
    cat[zero_mask] = 0
    cat[subnormal_mask] = 1
    cat[normal_mask] = 2
    cat[inf_mask] = 3
    cat[nan_mask] = 4

    return sign * 4 + cat

def grade(sol, fx) -> dict:
    tests = np.array([
        0x00000000,  # +0
        0x80000000,  # -0
        0x00400000,  # +subnormal
        0x80400000,  # -subnormal
        0x3F800000,  # +1.0 normal
        0xBF800000,  # -1.0 normal
        0x7F800000,  # +inf
        0xFF800000,  # -inf
        0x7FC00000   # NaN
    ], dtype=np.uint32)

    try:
        got = sol.classify_uint32_patterns(tests)
    except Exception:
        return {"exact_match": 0.0}

    ref = _oracle(tests)
    ok = float(np.array_equal(got, ref))
    return {"exact_match": ok}
