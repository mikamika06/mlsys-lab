import numpy as np


def _finite_difference_grads(A, B, G, eps=1e-6):
    def loss(a, b):
        return float(np.sum((a @ b) * G))

    dA = np.zeros_like(A, dtype=np.float64)
    dB = np.zeros_like(B, dtype=np.float64)

    for idx in np.ndindex(A.shape):
        plus = A.copy()
        minus = A.copy()
        plus[idx] += eps
        minus[idx] -= eps
        dA[idx] = (loss(plus, B) - loss(minus, B)) / (2 * eps)

    for idx in np.ndindex(B.shape):
        plus = B.copy()
        minus = B.copy()
        plus[idx] += eps
        minus[idx] -= eps
        dB[idx] = (loss(A, plus) - loss(A, minus)) / (2 * eps)

    return dA, dB


def grade(sol, fx) -> dict:
    cases = [
        (
            np.array([[1.0, -2.0, 0.5], [3.0, 4.0, -1.0]]),
            np.array([[2.0, 1.0], [-3.0, 5.0], [4.0, -2.0]]),
            np.array([[0.5, -1.5], [2.0, 3.0]]),
        ),
        (
            np.array([[0.2, 1.3], [-4.0, 2.5], [1.1, -0.7]]),
            np.array([[3.0, -2.0, 1.0], [0.5, 4.0, -3.0]]),
            np.array([[1.0, 2.0, -1.0], [0.5, -0.5, 3.0], [-2.0, 1.5, 0.0]]),
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
            float(np.max(np.abs(np.asarray(got_a) - ref_a))),
            float(np.max(np.abs(np.asarray(got_b) - ref_b))),
        )
        worst = max(worst, err)

    return {"max_abs_err": worst}
