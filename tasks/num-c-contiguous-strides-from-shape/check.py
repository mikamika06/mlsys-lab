def grade(sol, fx) -> dict:
    import numpy as np
    cases = [
        ((3,), np.int32),
        ((2, 5), np.float64),
        ((4, 3, 2), 'float32'),
        ((1, 10, 20), np.uint8),
        ((7, 6, 5, 4), np.int16)
    ]
    ok = 1.0
    for shape, dtype in cases:
        try:
            got = sol.c_contig_strides(shape, dtype)
            ref_arr = np.empty(shape, dtype=dtype)
            ref = tuple(ref_arr.strides)
        except Exception:
            return {"exact_match": 0.0}
        if got != ref:
            ok = 0.0
            break
    return {"exact_match": ok}
