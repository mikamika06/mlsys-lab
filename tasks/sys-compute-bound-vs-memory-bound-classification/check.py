import numpy as np

def grade(sol, fx) -> dict:
    # Test cases covering normal values and the boundary case
    cases = [
        (np.array([0.2, 0.8, 1.5]), 1.0),
        (np.array([1.0, 1.0, 1.0]), 1.0),   # boundary: all memory‑bound
        (np.array([10., 0.01]), 5.0),
    ]
    ok = 1.0
    for ai, bal in cases:
        try:
            got = sol.classify_bound(ai, bal)
        except Exception:
            return {"exact_match": 0.0}
        # Reference using the same rule
        ref = np.where(ai > bal, 'compute-bound', 'memory-bound')
        if not isinstance(got, (np.ndarray, list)):
            ok = 0.0
            break
        got_arr = np.array(got)
        if got_arr.shape != ref.shape:
            ok = 0.0
            break
        if not np.all(got_arr == ref):
            ok = 0.0
            break
    return {"exact_match": ok}
