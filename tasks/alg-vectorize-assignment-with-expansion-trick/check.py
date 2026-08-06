import numpy as np
import sys

def _reference_assign(X, centroids):
    X_norm = np.sum(X**2, axis=1)[:, None]
    C_norm = np.sum(centroids**2, axis=1)[None, :]
    cross = X @ centroids.T
    dists_sq = X_norm + C_norm - 2*cross
    return np.argmin(dists_sq, axis=1).tolist()

def _count_lines(func):
    counter = 0
    def trace(frame, event, arg):
        nonlocal counter
        if event == 'line':
            counter += 1
        return trace
    sys.settrace(trace)
    try:
        func()
    finally:
        sys.settrace(None)
    return counter

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    tests = [
        (rng.integers(0, 10, size=(5,3)), rng.integers(0, 10, size=(2,3))),
        (rng.normal(size=(20,4)), rng.normal(size=(3,4))),
        (rng.standard_normal(size=(7,1)), rng.standard_normal(size=(4,1))),
    ]
    exact = 1.0
    for X_np, C_np in tests:
        X = X_np.astype(np.float64).tolist()
        C = C_np.astype(np.float64).tolist()
        try:
            got = sol.assign_clusters(X, C)
        except Exception:
            return {"exact_match": 0.0, "op_count": float("inf")}
        ref = _reference_assign(X_np, C_np)
        if got != ref:
            exact = 0.0
            break
    X_np, C_np = tests[0]
    X = X_np.astype(np.float64).tolist()
    C = C_np.astype(np.float64).tolist()
    try:
        op_cnt = _count_lines(lambda: sol.assign_clusters(X, C))
    except Exception:
        op_cnt = float("inf")
    return {"exact_match": exact, "op_count": op_cnt}
