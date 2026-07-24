import numpy as np

def _oracle(L, T):
    M = np.zeros((L, T), dtype=int)
    for t in range(T):
        i = t % L
        M[i, t] = 1
        M[(i + 1) % L, t] = 1
    return M

def grade(sol, fx) -> dict:
    cases = [
        (4, 6),
        (3, 5),
        (7, 10),
        (2, 8),
        (5, 12)
    ]
    for L, T in cases:
        try:
            got = sol.label_gpu_residency(L, T)
            ref = _oracle(L, T)
        except Exception:
            return {"exact_match": 0.0}
        if not isinstance(got, np.ndarray):
            return {"exact_match": 0.0}
        if got.shape != ref.shape or got.dtype.kind not in ('i', 'b'):
            return {"exact_match": 0.0}
        if not np.array_equal(got, ref):
            return {"exact_match": 0.0}
    return {"exact_match": 1.0}
