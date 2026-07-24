import numpy as np


def _oracle(x, base, A, B, scale):
    x = np.asarray(x, dtype=np.float64)
    base = np.asarray(base, dtype=np.float64)
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    return base + scale * (x @ A) @ B


def grade(sol, fx) -> dict:
    x, base, A, B = fx["x"], fx["base"], fx["a"], fx["b"]

    max_err = 0.0
    for scale in (2.0, 0.0, -1.5, 8.0):
        ref = _oracle(x, base, A, B, scale)
        try:
            got = sol.lora_delta_forward(x.copy(), base.copy(), A.copy(), B.copy(), scale)
            got = np.asarray(got, dtype=np.float64)
        except Exception:
            return {"max_abs_err": float("inf")}

        if got.shape != ref.shape:
            return {"max_abs_err": float("inf")}

        max_err = max(max_err, float(np.max(np.abs(got - ref))))

    return {"max_abs_err": max_err}
