import numpy as np

def _ref(A):
    amax = np.max(np.abs(A), axis=1)
    scales = np.where(amax == 0, 1.0, amax / 127.0)
    codes = np.round(A / scales[:, None]).clip(-128, 127).astype(np.int8)
    return codes, scales

def grade(sol, fx) -> dict:
    cases = [
        (np.array([[0, -2, 3], [4, 0, -5]], dtype=np.float32),),
        (np.random.randn(5, 7).astype(np.float64),),
        (np.zeros((3, 4)),),
        (np.random.uniform(-10, 10, size=(8, 6)).astype(np.float32),),
    ]
    ok = 1.0
    for (A,) in cases:
        try:
            got_codes, got_scales = sol.per_token_int8_quant(A)
        except Exception:
            return {"exact_match": 0.0}
        ref_codes, ref_scales = _ref(A)
        if not (np.array_equal(got_codes, ref_codes) and np.allclose(got_scales, ref_scales)):
            ok = 0.0
            break
    return {"exact_match": ok}
