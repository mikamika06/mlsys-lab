import numpy as np


def _oracle_lu(A):
    A = np.asarray(A, dtype=np.float64)
    n = A.shape[0]
    M = A.copy()
    P = np.eye(n, dtype=np.float64)
    L = np.eye(n, dtype=np.float64)

    for k in range(n - 1):
        pivot = k + int(np.argmax(np.abs(M[k:, k])))

        if pivot != k:
            M[[k, pivot], :] = M[[pivot, k], :]
            P[[k, pivot], :] = P[[pivot, k], :]
            if k > 0:
                L[[k, pivot], :k] = L[[pivot, k], :k]

        for i in range(k + 1, n):
            L[i, k] = M[i, k] / M[k, k]
            M[i, k:] -= L[i, k] * M[k, k:]

    return P, L, M


def grade(sol, fx) -> dict:
    cases = [
        np.array([[0.0, 1.0], [1.0, 1.0]]),
        np.array([[0.0, 2.0, 1.0], [1.0, 1.0, 3.0], [2.0, 4.0, 0.0]]),
        np.array([[5.0, 2.0, -1.0], [1.0, 3.0, 4.0], [2.0, 0.0, 1.0]]),
        np.array([[1.0, 2.0, 3.0, 4.0],
                  [5.0, 6.0, 7.0, 8.0],
                  [2.0, 1.0, 0.0, 3.0],
                  [4.0, 2.0, 1.0, 5.0]]),
    ]

    worst = 0.0
    for A in cases:
        try:
            ref_P, ref_L, ref_U = _oracle_lu(A)

            P, L, U = sol.lu_partial_pivot(A.copy())
            P = np.asarray(P, dtype=np.float64)
            L = np.asarray(L, dtype=np.float64)
            U = np.asarray(U, dtype=np.float64)

            err = max(
                float(np.max(np.abs(P - ref_P))),
                float(np.max(np.abs(L - ref_L))),
                float(np.max(np.abs(U - ref_U))),
                float(np.max(np.abs(P @ A - L @ U))),
            )
            if not np.isfinite(err):
                return {"max_abs_err": float("inf")}
            worst = max(worst, err)
        except Exception:
            return {"max_abs_err": float("inf")}

    return {"max_abs_err": worst}
