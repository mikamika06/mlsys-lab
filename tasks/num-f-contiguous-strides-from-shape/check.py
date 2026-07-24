import numpy as np

def grade(sol, fx) -> dict:
    cases = [
        (),
        (3,),
        (2, 5),
        (4, 6, 8),
        (1, 2, 3, 4)
    ]
    ok = 1.0
    for shape in cases:
        try:
            got = sol.f_contiguous_strides_from_shape(shape)
        except Exception:
            return {"exact_match": 0.0}
        ref_arr = np.zeros(shape, dtype=np.float64, order='F')
        ref = tuple(ref_arr.strides)
        if got != ref:
            ok = 0.0
            break
    return {"exact_match": ok}
