import numpy as np

def grade(sol, fx) -> dict:
    # Generate a handful of test cases with scalar and array inputs
    cases = [
        (np.array([1.0, 2.0]), np.array([10.0, 20.0]), np.array([5.0, 4.0])),
        (1.5, 12.0, 3.0),
        (np.arange(5), np.full(5, 100.0), np.linspace(2.0, 6.0, 5))
    ]
    errors = []
    for ai, peak, bw in cases:
        try:
            got = sol.roofline_perf(ai, peak, bw)
        except Exception:
            return {"rel_err": 1.0}
        ref = np.minimum(peak, ai * bw)
        err = np.linalg.norm(got - ref) / (np.linalg.norm(ref) + 1e-12)
        errors.append(err)
    return {"rel_err": max(errors)}
