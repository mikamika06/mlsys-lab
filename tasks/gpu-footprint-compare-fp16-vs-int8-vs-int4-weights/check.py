import numpy as np

def _expected_ratios(fp16, i8c, i8s, i4c, i4s):
    fp_bytes = fp16.nbytes
    int8_total = i8c.nbytes + i8s.nbytes
    int4_total = i4c.nbytes + i4s.nbytes
    return fp_bytes / int8_total, fp_bytes / int4_total

def _make_case(shape):
    rng = np.random.default_rng(0)
    n, d = shape
    fp16 = rng.standard_normal((n, d)).astype(np.float16)

    # INT8 codes same shape, signed byte
    i8c = fp16.astype(np.int8)

    # One scale per column (channel)
    i8s = rng.random(d).astype(np.float32)

    # Packed INT4: two 4‑bit values per uint8, so half the columns
    d4 = (d + 1) // 2
    i4c = rng.integers(0, 256, size=(n, d4), dtype=np.uint8)
    i4s = rng.random(d).astype(np.float32)

    return fp16, i8c, i8s, i4c, i4s

def grade(sol, fx) -> dict:
    # Test multiple shapes
    cases = [
        (64, 128),
        (32, 63),
        (1, 10),
        (100, 200),
        (50, 37)
    ]
    ok_int8 = 1.0
    ok_int4 = 1.0

    for shape in cases:
        fp16, i8c, i8s, i4c, i4s = _make_case(shape)

        try:
            got_int8, got_int4 = sol.size_ratio_fp16_quantized(
                fp16, i8c, i8s, i4c, i4s
            )
        except Exception:
            return {"int8_size_ratio": 0.0, "int4_size_ratio": 0.0}

        exp_int8, exp_int4 = _expected_ratios(fp16, i8c, i8s, i4c, i4s)

        if not np.isclose(got_int8, exp_int8, rtol=1e-12, atol=1e-15):
            ok_int8 = 0.0
        if not np.isclose(got_int4, exp_int4, rtol=1e-12, atol=1e-15):
            ok_int4 = 0.0

    return {"int8_size_ratio": ok_int8, "int4_size_ratio": ok_int4}
