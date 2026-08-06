import numpy as np


def _finite_difference_grads(A, B, G, eps=1e-6):
    def loss(a, b):
        m = len(a)
        k = len(a[0])
        n = len(b[0])
        ab = [[sum(a[i][p] * b[p][j] for p in range(k)) for j in range(n)] for i in range(m)]
        return float(sum(ab[i][j] * G[i][j] for i in range(m) for j in range(n)))

    m_a, k_a = len(A), len(A[0])
    k_b, n_b = len(B), len(B[0])

    dA = [[0.0] * k_a for _ in range(m_a)]
    dB = [[0.0] * n_b for _ in range(k_b)]

    for r in range(m_a):
        for c in range(k_a):
            plus = [row[:] for row in A]
            minus = [row[:] for row in A]
            plus[r][c] += eps
            minus[r][c] -= eps
            dA[r][c] = (loss(plus, B) - loss(minus, B)) / (2 * eps)

    for r in range(k_b):
        for c in range(n_b):
            plus = [row[:] for row in B]
            minus = [row[:] for row in B]
            plus[r][c] += eps
            minus[r][c] -= eps
            dB[r][c] = (loss(A, plus) - loss(A, minus)) / (2 * eps)

    return dA, dB


def grade(sol, fx) -> dict:
    cases = [
        (
            [[1.0, -2.0, 0.5], [3.0, 4.0, -1.0]],
            [[2.0, 1.0], [-3.0, 5.0], [4.0, -2.0]],
            [[0.5, -1.5], [2.0, 3.0]],
        ),
        (
            [[0.2, 1.3], [-4.0, 2.5], [1.1, -0.7]],
            [[3.0, -2.0, 1.0], [0.5, 4.0, -3.0]],
            [[1.0, 2.0, -1.0], [0.5, -0.5, 3.0], [-2.0, 1.5, 0.0]],
        ),
    ]

    worst = 0.0
    for A, B, G in cases:
        try:
            got_a, got_b = sol.matmul_vjp(A, B, G)
        except Exception:
            return {"max_abs_err": float("inf")}

        ref_a, ref_b = _finite_difference_grads(A, B, G)
        err = max(
            float(np.max(np.abs(np.asarray(got_a) - np.asarray(ref_a)))),
            float(np.max(np.abs(np.asarray(got_b) - np.asarray(ref_b)))),
        )
        worst = max(worst, err)

    return {"max_abs_err": worst}
