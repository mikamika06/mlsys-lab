import numpy as np
from scipy.linalg import lu_factor


def _oracle_piv(A: np.ndarray) -> np.ndarray:
    _, piv = lu_factor(np.asarray(A, dtype=np.float64))
    return np.asarray(piv, dtype=np.int64)


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    cases = [np.array([
        [2.0, 1.0, 1.0],
        [4.0, 3.0, 3.0],
        [8.0, 7.0, 9.0],
    ])]
    for _ in range(40):
        n = int(rng.integers(2, 9))
        cases.append(rng.standard_normal((n, n)) * 10.0)

    n_ok = 0
    for A in cases:
        ref = _oracle_piv(A)
        try:
            got = np.asarray(sol.lu_pivot_indices(A.copy()), dtype=np.int64)
        except Exception:
            continue
        if got.shape == ref.shape and np.array_equal(got, ref):
            n_ok += 1

    return {"exact_match": float(n_ok / len(cases))}
