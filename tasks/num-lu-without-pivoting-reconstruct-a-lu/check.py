import numpy as np


def _reference_lu(A):
    A = np.asarray(A, dtype=np.float64)
    n = A.shape[0]
    L = np.eye(n, dtype=np.float64)
    U = np.zeros((n, n), dtype=np.float64)

    for k in range(n):
        for j in range(k, n):
            U[k, j] = A[k, j] - np.sum(L[k, :k] * U[:k, j])

        for i in range(k + 1, n):
            L[i, k] = (
                A[i, k] - np.sum(L[i, :k] * U[:k, k])
            ) / U[k, k]

    return L, U


def grade(sol, fx) -> dict:
    cases = [
        np.array([
            [4.0, 3.0],
            [6.0, 3.0],
        ]),
        np.array([
            [10.0, 2.0, 3.0],
            [3.0, 8.0, 1.0],
            [2.0, 4.0, 9.0],
        ]),
        np.array([
            [5.0, -1.0, 2.0, 0.5],
            [2.0, 7.0, 1.0, -2.0],
            [3.0, 0.0, 6.0, 1.0],
            [1.0, 4.0, -1.0, 8.0],
        ]),
    ]

    worst = 0.0
    for A in cases:
        try:
            L, U = sol.lu_no_pivot(A.copy())
            L = np.asarray(L, dtype=np.float64)
            U = np.asarray(U, dtype=np.float64)

            ref_L, ref_U = _reference_lu(A)

            err = max(
                float(np.max(np.abs(L - ref_L))),
                float(np.max(np.abs(U - ref_U))),
                float(np.max(np.abs(L @ U - A))),
            )
            worst = max(worst, err)
        except Exception:
            worst = float("inf")
            break

    return {"max_abs_err": worst}
