import numpy as np


def _oracle(A):
    return float(np.linalg.det(np.asarray(A, dtype=np.float64)))


def grade(sol, fx) -> dict:
    cases = [
        np.array([[2.0, 1.0], [4.0, 3.0]]),
        np.array([[0.0, 2.0, 1.0], [1.0, 0.0, 3.0], [4.0, 1.0, 2.0]]),
        np.array([[3.0, -1.0, 2.0, 4.0],
                  [1.0, 5.0, 0.0, 2.0],
                  [2.0, 1.0, 3.0, -1.0],
                  [0.0, 2.0, 1.0, 1.0]]),
        np.array([[1.0, 2.0, 3.0],
                  [2.0, 4.0, 6.0],
                  [0.0, 1.0, 1.0]]),
    ]

    refs = [_oracle(x) for x in cases]
    vals = []
    try:
        for x in cases:
            vals.append(float(sol.det_from_lu(x.copy())))
    except Exception:
        return {"rel_err": float("inf")}

    num = np.linalg.norm(np.asarray(vals) - np.asarray(refs))
    den = np.linalg.norm(np.asarray(refs)) + 1e-12
    return {"rel_err": float(num / den)}
