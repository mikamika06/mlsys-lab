import numpy as np


def _oracle(A, B, bias, activation):
    x = A @ B + bias
    if activation == "relu":
        return np.maximum(x, 0.0)
    if activation == "identity":
        return x
    raise ValueError("unknown activation")


def grade(sol, fx) -> dict:
    cases = [
        (
            np.array([[1.0, -2.0], [3.0, 4.0]], dtype=np.float64),
            np.array([[2.0, -1.0, 0.5], [1.0, 3.0, -2.0]], dtype=np.float64),
            np.array([0.5, -1.0, 2.0], dtype=np.float64),
            "relu",
        ),
        (
            np.array([[0.2, -0.7, 1.5]], dtype=np.float64),
            np.array([[1.0, 2.0], [-3.0, 0.5], [2.0, -1.0]], dtype=np.float64),
            np.array([0.0, 1.0], dtype=np.float64),
            "identity",
        ),
    ]

    max_err = 0.0
    reuse = 1.0

    for A, B, bias, activation in cases:
        out = np.full((A.shape[0], B.shape[1]), np.nan, dtype=np.float64)
        try:
            got = sol.fused_matmul_epilogue(
                A.copy(), B.copy(), bias.copy(), activation, out
            )
        except Exception:
            return {"max_abs_err": float("inf"), "out_reuse": 0.0}

        if got is not out:
            reuse = 0.0

        ref = _oracle(A, B, bias, activation)
        try:
            err = float(np.max(np.abs(np.asarray(got) - ref)))
        except Exception:
            err = float("inf")
        max_err = max(max_err, err)

    return {
        "max_abs_err": max_err,
        "out_reuse": reuse,
    }
