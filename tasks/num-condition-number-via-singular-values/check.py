def grade(sol, fx) -> dict:
    """Grade condition_number_via_svd against NumPy's oracle."""
    import numpy as np

    rng = np.random.RandomState(42)

    # Diverse test matrices: well-conditioned, ill-conditioned, near-singular,
    # random, and nearly rank-deficient.
    cases = [
        np.eye(3),
        np.diag([2.0, 3.0, 5.0]),
        np.array([[1.0, 2.0],
                  [3.0, 4.0]]),
        np.diag([1e10, 1e-10]),
        np.diag([1.0, 1e-15]),
        rng.randn(5, 5),
        rng.randn(6, 6),
        np.array([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0],
                  [7.0, 8.0, 9.0]]),  # near rank-deficient
        np.diag([1.0, 1.0, 1.0, 1e-12]),
        rng.randn(8, 8),
    ]

    max_rel_err = 0.0
    for A in cases:
        try:
            got = float(sol.condition_number_via_svd(A))
            ref = float(np.linalg.cond(A))
            if ref == 0.0:
                err = 0.0 if got == 0.0 else 1.0
            else:
                err = abs(got - ref) / (abs(ref) + 1e-300)
            max_rel_err = max(max_rel_err, err)
        except Exception:
            # Any exception means total failure on this case
            max_rel_err = 1.0
            break

    return {"rel_err": max_rel_err}
