import numpy as np

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    n, d, m, k, t = 200, 50, 20, 5, 10
    A = rng.standard_normal((n, d))
    Q = rng.standard_normal((m, d))
    try:
        recall = float(sol.lsh_recall(A, Q, k, t, seed=42))
    except Exception:
        return {"recall": 0.0}
    if not isinstance(recall, float):
        return {"recall": 0.0}
    return {"recall": recall}
