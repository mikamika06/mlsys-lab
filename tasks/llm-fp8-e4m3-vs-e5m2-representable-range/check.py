import numpy as np

def _reference(values):
    values = np.asarray(values, dtype=np.float64)
    abs_vals = np.abs(values)
    finite_mask = np.isfinite(values)

    # e4m3 parameters
    m1, e1 = 3, 4
    bias1 = (1 << (e1 - 1)) - 1          # 7
    max_exp_field1 = (1 << e1) - 2       # 14
    max_exponent1 = max_exp_field1 - bias1
    min_normal1 = 2 ** (1 - bias1)
    min_subnormal1 = 2 ** (1 - bias1 - m1)
    max_finite1 = (2 - 2 ** (-m1)) * 2 ** max_exponent1

    mask_e4m3 = finite_mask & (
        (abs_vals == 0) |
        ((abs_vals >= min_subnormal1) & (abs_vals <= max_finite1))
    )

    # e5m2 parameters
    m2, e2 = 2, 5
    bias2 = (1 << (e2 - 1)) - 1          # 15
    max_exp_field2 = (1 << e2) - 2       # 30
    max_exponent2 = max_exp_field2 - bias2
    min_normal2 = 2 ** (1 - bias2)
    min_subnormal2 = 2 ** (1 - bias2 - m2)
    max_finite2 = (2 - 2 ** (-m2)) * 2 ** max_exponent2

    mask_e5m2 = finite_mask & (
        (abs_vals == 0) |
        ((abs_vals >= min_subnormal2) & (abs_vals <= max_finite2))
    )

    return mask_e4m3, mask_e5m2


def grade(sol, fx) -> dict:
    # Test cases covering boundaries and random values
    rng = np.random.default_rng(0)
    tests = []

    # Edge values around subnormal limits
    for m in [1e-9, 1e-4, 1e-3, 1e-2]:
        tests.append(np.array([m]))
    # Max finite and just above
    tests.append(np.array([240.0, 241.0]))
    tests.append(np.array([57344.0, 60000.0]))

    # Random values across wide range
    for _ in range(5):
        vals = rng.uniform(-1e6, 1e6, size=20)
        tests.append(vals)

    # Special values
    tests.append(np.array([0.0, -0.0, np.nan, np.inf, -np.inf]))

    ok = 1.0
    for case in tests:
        try:
            got_e4m3, got_e5m2 = sol.fp8_representability(case)
            ref_e4m3, ref_e5m2 = _reference(case)
        except Exception:
            return {"exact_match": 0.0}
        if not (np.array_equal(got_e4m3, ref_e4m3) and np.array_equal(got_e5m2, ref_e5m2)):
            ok = 0.0
            break

    return {"exact_match": ok}
