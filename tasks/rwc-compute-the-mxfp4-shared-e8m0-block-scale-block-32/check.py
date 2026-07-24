def grade(sol, fx) -> dict:
    import numpy as np
    ok = 1.0
    rng = np.random.default_rng(12345)
    for _ in range(5):
        B = rng.integers(1, 10)
        # generate random weights with values up to ~10
        weights = rng.standard_normal((B, 32)).astype(np.float64) * rng.uniform(0, 10, size=(B, 1))
        amax = np.max(np.abs(weights), axis=1)
        ref = np.maximum(0, np.ceil(np.log2(amax / 6.0))).astype(np.int32)
        try:
            got = sol.compute_shared_e8m0_scale(weights)
            got_arr = np.asarray(got, dtype=np.int32).flatten()
        except Exception:
            ok = 0.0
            break
        if got_arr.shape != ref.shape or not np.array_equal(got_arr, ref):
            ok = 0.0
            break
    return {"exact_match": ok}
