import numpy as np

def _ref_quantize(weights):
    w = np.asarray(weights, dtype=np.float64)
    n = len(w)
    assert n % 32 == 0, "Length must be multiple of 32"
    codes = np.empty(n, dtype=np.uint8)
    scales = np.empty(n // 32, dtype=np.float16)
    for i in range(0, n, 32):
        block = w[i:i+32]
        d = np.max(np.abs(block)) / 8.0
        if d == 0:
            d = 1e-12
        c = np.clip(np.round(block / d).astype(int) + 8, 0, 15)
        codes[i:i+32] = c.astype(np.uint8)
        scales[i // 32] = d
    return codes, scales

def _ref_dequantize(codes, scales):
    c = np.asarray(codes, dtype=np.int16)
    n = len(c)
    assert n % 32 == 0, "Length must be multiple of 32"
    w_hat = np.empty(n, dtype=np.float64)
    for i in range(0, n, 32):
        block_c = c[i:i+32] - 8
        d = scales[i // 32]
        w_hat[i:i+32] = block_c * d
    return w_hat

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    cases = [
        rng.standard_normal(64),
        rng.standard_normal(96),
        rng.integers(-100, 101, size=128).astype(float),
    ]
    ok = 1.0
    for weights in cases:
        try:
            ref_codes, ref_scales = _ref_quantize(weights)
            got_codes, got_scales = sol.q4_0_quantize(weights)
            if not np.array_equal(got_codes, ref_codes):
                ok = 0.0
                break
            recovered = sol.q4_0_dequantize(got_codes, got_scales)
            a = np.asarray(weights, dtype=np.float64).ravel()
            b = np.asarray(recovered, dtype=np.float64).ravel()
            rel_err = float(np.linalg.norm(b - a) / (np.linalg.norm(a) + 1e-12))
            if rel_err > 0.5:
                ok = 0.0
                break
        except Exception:
            ok = 0.0
            break
    return {"exact_match": ok}
