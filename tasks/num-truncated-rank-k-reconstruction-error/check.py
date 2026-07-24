import numpy as np


def _oracle(A, k):
    _, s, _ = np.linalg.svd(np.asarray(A, dtype=np.float64), full_matrices=False)
    return float(s[k])


def grade(sol, fx) -> dict:
    cases = [
        (np.array([[3.0, 0.0], [0.0, 2.0], [0.0, 1.0]]), 1),
        (np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), 1),
        (np.arange(20, dtype=np.float64).reshape(4, 5), 2),
        (np.array([[2.0, -1.0], [4.0, 3.0], [1.0, 8.0], [0.0, 5.0]]), 1),
        (np.eye(5, dtype=np.float64), 3),
    ]

    worst = 0.0
    for A, k in cases:
        try:
            got = float(sol.truncated_rank_k_error(A.copy(), k))
        except Exception:
            return {"rel_err": float("inf")}

        ref = _oracle(A, k)
        err = abs(got - ref) / (abs(ref) + 1e-12)
        worst = max(worst, err)

    return {"rel_err": worst}
