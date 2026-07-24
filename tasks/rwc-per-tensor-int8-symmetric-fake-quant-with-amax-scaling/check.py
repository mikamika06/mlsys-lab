import numpy as np

def _ref(x):
    x = np.asarray(x)
    amax = np.max(np.abs(x))
    if amax == 0:
        scale = 1.0
    else:
        scale = amax / 127.0
    q = np.round(x / scale)
    q_clipped = np.clip(q, -127, 127).astype(np.int8)
    dq = q_clipped.astype(np.float64) * scale
    return q_clipped, dq

def grade(sol, fx):
    rng = np.random.default_rng(0)
    cases = [
        rng.standard_normal((3,)),
        rng.standard_normal((4,5)) * 10,
        np.zeros((7,)),
        rng.uniform(-1000, 1000, size=(6,)),
        rng.standard_normal((2,3)) * 5
    ]

    ok_codes = True
    max_err = 0.0

    for x in cases:
        try:
            codes, dq = sol.per_tensor_int8_symmetric_fake_quant(x)
        except Exception:
            return {"exact_codes": 0.0, "max_abs_err": float("inf")}

        ref_codes, ref_dq = _ref(x)

        if not np.array_equal(codes, ref_codes):
            ok_codes = False

        err = np.max(np.abs(dq - ref_dq))
        if err > max_err:
            max_err = err

    return {"exact_codes": 1.0 if ok_codes else 0.0,
            "max_abs_err": max_err}
