def grade(sol, fx) -> dict:
    import numpy as np
    from mlsys import scorers

    cases = [
        (np.random.randn(1000), 64, 32, 8),
        (np.random.randn(12345), 128, 64, 16),
        (np.array([1.0]*5000), 256, 128, 32),
        (np.arange(2000).reshape(-1,1), 512, 256, 64)
    ]

    errors = []
    for weights, bs, ob, ib in cases:
        try:
            got = sol.compute_nf4_bits(weights, bs, ob, ib)
            if not isinstance(got, tuple) or len(got) != 2:
                return {"max_abs_err": float("inf")}
            ref_no_double = 4 + 32 / bs
            ref_double = 4 + 8 / ob + 32 / (ob * ib)
            err = scorers.max_abs_err(np.array(got), np.array([ref_no_double, ref_double]))
        except Exception:
            return {"max_abs_err": float("inf")}
        errors.append(err)

    return {"max_abs_err": max(errors)}
