import numpy as np

# ---- Oracle (built from the grid, never hardcoded expected outputs) ----

_MAG = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])

def _oracle(x):
    """Snap to nearest signed E2M1 level using pure NumPy."""
    x = np.asarray(x, dtype=np.float64)
    abs_x = np.abs(x)
    # (n,1) - (1,8) broadcast
    diffs = np.abs(abs_x[:, np.newaxis] - _MAG[np.newaxis, :])
    idx = np.argmin(diffs, axis=1)
    return np.sign(x) * _MAG[idx]

# ---- Grader ----

def grade(sol, fx) -> dict:
    tests = [
        # midpoints between grid levels
        np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]),
        # negatives
        np.array([-0.1, -0.3, -0.6, -1.1, -2.8, -5.5]),
        # exact grid values
        np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]),
        # near-boundary tie (0.25 is equidistant to 0 and 0.5)
        np.array([0.25]),
        # zeros
        np.array([0.0]),
        # large positives and negatives (near clamp)
        np.array([-6.49, -6.01, 6.01, 6.49, -6.5, 6.5]),
        # mixed extremes
        np.array([-0.01, 0.01, -5.99, 5.99, -6.499, 6.499]),
    ]

    worst = 0.0
    for x in tests:
        x = np.asarray(x, dtype=np.float64)
        try:
            got = np.asarray(sol.snap_to_e2m1(x), dtype=np.float64).ravel()
        except Exception:
            return {"max_abs_err": float("inf")}
        if got.shape != x.shape:
            return {"max_abs_err": float("inf")}
        ref = _oracle(x)
        err = float(np.max(np.abs(got - ref)))
        if err > worst:
            worst = err
    return {"max_abs_err": worst}
