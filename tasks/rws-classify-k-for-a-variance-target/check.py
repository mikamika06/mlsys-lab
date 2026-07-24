import numpy as np

def grade(sol, fx) -> dict:
    # Test cases: (eigenvalues array, target)
    cases = [
        (np.array([4.0, 2.0, 1.0, 0.5]), 0.8),
        (np.array([3.0, 3.0, 3.0]), 0.5),
        (np.array([10.0, 5.0, 2.0, 1.0]), 0.9),
        (np.arange(1, 11), 0.7),
        (np.random.rand(20), 0.6)
    ]
    ok = 1.0
    for eigs, target in cases:
        try:
            got = sol.classify_k_for_variance_target(eigs, target)
        except Exception:
            ok = 0.0
            break
        total = np.sum(eigs)
        cum = np.cumsum(eigs)
        ratio = cum / total
        expected = int(np.searchsorted(ratio, target, side='left') + 1)
        if got != expected:
            ok = 0.0
            break
    return {"argmin_index": ok}
