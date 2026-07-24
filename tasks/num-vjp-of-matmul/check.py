import numpy as np
from mlsys.scorers import max_abs_err

def _finite_diff_vjp(A, B, dY, eps=1e-6):
    """Compute VJP of A @ B via central finite differences."""
    m, k = A.shape
    k2, n = B.shape
    assert k == k2
    dA = np.zeros_like(A)
    dB = np.zeros_like(B)

    # Gradient w.r.t. A
    it = np.nditer(A, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        A_plus = A.copy()
        A_minus = A.copy()
        A_plus[idx] += eps
        A_minus[idx] -= eps
        Y_plus = A_plus @ B
        Y_minus = A_minus @ B
        grad = np.sum((Y_plus - Y_minus) * dY) / (2 * eps)
        dA[idx] = grad
        it.iternext()

    # Gradient w.r.t. B
    it = np.nditer(B, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        B_plus = B.copy()
        B_minus = B.copy()
        B_plus[idx] += eps
        B_minus[idx] -= eps
        Y_plus = A @ B_plus
        Y_minus = A @ B_minus
        grad = np.sum((Y_plus - Y_minus) * dY) / (2 * eps)
        dB[idx] = grad
        it.iternext()

    return dA, dB

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    cases = [
        (rng.standard_normal((3, 4)), rng.standard_normal((4, 5))),
        (rng.standard_normal((5, 6)), rng.standard_normal((6, 7))),
        (rng.standard_normal((2, 2)), rng.standard_normal((2, 3))),
    ]

    overall_max = 0.0
    for A, B in cases:
        m, k = A.shape
        _, n = B.shape
        dY = rng.standard_normal((m, n))
        try:
            cand_dA, cand_dB = sol.vjp_matmul(A, B, dY)
        except Exception:
            return {"max_abs_err": float("inf")}

        ref_dA, ref_dB = _finite_diff_vjp(A, B, dY)

        err_A = max_abs_err(cand_dA, ref_dA)
        err_B = max_abs_err(cand_dB, ref_dB)
        overall_max = max(overall_max, err_A, err_B)

    return {"max_abs_err": overall_max}
