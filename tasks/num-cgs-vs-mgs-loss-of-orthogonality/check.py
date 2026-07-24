import numpy as np


def _loss(Q):
    n = Q.shape[1]
    return float(np.max(np.abs(Q.T @ Q - np.eye(n, dtype=np.float64))))


def _ref_cgs(A):
    A = np.asarray(A, dtype=np.float64)
    m, n = A.shape
    Q = np.zeros((m, n), dtype=np.float64)
    for j in range(n):
        v = A[:, j].copy()
        for i in range(j):
            v -= np.dot(Q[:, i], A[:, j]) * Q[:, i]
        Q[:, j] = v / np.linalg.norm(v)
    return _loss(Q)


def _ref_mgs(A):
    A = np.asarray(A, dtype=np.float64)
    m, n = A.shape
    V = A.copy()
    Q = np.zeros((m, n), dtype=np.float64)
    for i in range(n):
        q = V[:, i] / np.linalg.norm(V[:, i])
        Q[:, i] = q
        for j in range(i + 1, n):
            V[:, j] -= np.dot(q, V[:, j]) * q
    return _loss(Q)


def grade(sol, fx) -> dict:
    A = np.array(
        [[1.0 / (i + j + 1) for j in range(6)] for i in range(6)],
        dtype=np.float64,
    )

    ref = np.array([_ref_cgs(A), _ref_mgs(A)], dtype=np.float64)

    try:
        got = np.asarray(sol.gram_schmidt_orthogonality(A), dtype=np.float64)
    except Exception:
        return {
            "max_abs_err": 1.0,
            "mgs_orthogonality": 1.0,
        }

    if got.shape != (2,) or not np.all(np.isfinite(got)):
        return {
            "max_abs_err": 1.0,
            "mgs_orthogonality": 1.0,
        }

    return {
        "max_abs_err": float(np.max(np.abs(got - ref))),
        "mgs_orthogonality": float(got[1]),
    }
