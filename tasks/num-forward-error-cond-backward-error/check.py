import numpy as np


def _oracle(A, b, delta_b):
    x = np.linalg.solve(A, b)
    x_hat = np.linalg.solve(A, b + delta_b)

    forward_error = np.linalg.norm(x_hat - x) / (np.linalg.norm(x) + 1e-12)
    backward_error = np.linalg.norm(delta_b) / (np.linalg.norm(b) + 1e-12)
    bound = np.linalg.cond(A) * backward_error

    return float(forward_error), float(bound)


def grade(sol, fx) -> dict:
    cases = [
        (
            np.array([[3.0, 1.0], [1.0, 2.0]]),
            np.array([9.0, 8.0]),
            np.array([1e-8, -2e-8]),
        ),
        (
            np.array([[1.0, 0.2], [0.2, 1.5]]),
            np.array([1.0, 2.0]),
            np.array([1e-7, -1e-7]),
        ),
        (
            np.array([[10.0, 2.0], [3.0, 8.0]]),
            np.array([4.0, 7.0]),
            np.array([-2e-8, 3e-8]),
        ),
    ]

    max_rel = 0.0
    bound_ok = 1.0

    for A, b, delta_b in cases:
        ref_forward, ref_bound = _oracle(A, b, delta_b)

        try:
            got_forward, got_bound = sol.forward_error_bound(
                A.copy(), b.copy(), delta_b.copy()
            )
            got_forward = float(got_forward)
            got_bound = float(got_bound)
        except Exception:
            return {"bound_ok": 0.0, "rel_err": 1.0}

        max_rel = max(
            max_rel,
            abs(got_forward - ref_forward) / (abs(ref_forward) + 1e-12),
            abs(got_bound - ref_bound) / (abs(ref_bound) + 1e-12),
        )

        if got_forward > got_bound * (1.0 + 1e-10) + 1e-12:
            bound_ok = 0.0

    return {"bound_ok": bound_ok, "rel_err": max_rel}
