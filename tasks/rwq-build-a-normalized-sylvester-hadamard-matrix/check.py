import numpy as np


def _oracle_hadamard(n):
    h = np.array([[1.0]], dtype=np.float64)
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h / np.sqrt(n)


def grade(sol, fx) -> dict:
    cases = [1, 2, 4, 8, 16]
    max_err = 0.0
    max_orth_err = 0.0

    for n in cases:
        ref = _oracle_hadamard(n)
        try:
            got = np.asarray(sol.normalized_hadamard(n), dtype=np.float64)
        except Exception:
            return {
                "max_abs_err": float("inf"),
                "orthogonality_err": float("inf")
            }

        if got.shape != ref.shape:
            return {
                "max_abs_err": float("inf"),
                "orthogonality_err": float("inf")
            }

        max_err = max(max_err, float(np.max(np.abs(got - ref))))
        ident = np.eye(n, dtype=np.float64)
        max_orth_err = max(
            max_orth_err,
            float(np.max(np.abs(got @ got.T - ident)))
        )

    return {
        "max_abs_err": max_err,
        "orthogonality_err": max_orth_err
    }
