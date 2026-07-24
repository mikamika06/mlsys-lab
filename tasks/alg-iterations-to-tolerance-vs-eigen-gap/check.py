import numpy as np

def _ref(A, tol):
    n = A.shape[0]
    v = np.ones(n, dtype=np.float64)
    v /= np.linalg.norm(v)
    for i in range(1, 10001):
        w = A @ v
        norm_w = np.linalg.norm(w)
        if norm_w == 0:
            return i
        v_next = w / norm_w
        diff = np.linalg.norm(v_next - v)
        if diff < tol:
            return i
        v = v_next
    return 10000

def grade(sol, fx) -> dict:
    rng = np.random.RandomState(42)
    cases = [
        (np.eye(3), 1e-6),
        (np.diag([5., 3., 2.]), 1e-8),
        (np.array([[10., 0.], [0., 9.]]), 1e-7),
        ((rng.randn(5,5) + rng.randn(5,5).T)/2, 1e-6)
    ]
    for A, tol in cases:
        try:
            got = sol.iterations_to_tolerance(A, tol)
        except Exception:
            return {"exact_match": 0.0}
        ref = _ref(A, tol)
        if got != ref:
            return {"exact_match": 0.0}
    return {"exact_match": 1.0}
