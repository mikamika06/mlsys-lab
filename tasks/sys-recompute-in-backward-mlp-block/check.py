import numpy as np
from mlsys.scorers import max_abs_err

def _ref_forward(x, W1, b1, W2, b2):
    z = W1 @ x + b1
    a = np.maximum(0, z)
    return W2 @ a + b2

def _ref_backward(dy, x, W1, b1, W2, b2):
    z = W1 @ x + b1
    mask = (z > 0).astype(float)
    da = W2.T @ dy
    dz = da * mask
    return W1.T @ dz

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(42)
    err_out = 0.0
    err_grad = 0.0
    for _ in range(5):
        d = rng.integers(2, 6)
        h = rng.integers(2, 6)
        o = rng.integers(2, 6)
        x = rng.standard_normal(d)
        W1 = rng.standard_normal((h, d))
        b1 = rng.standard_normal(h)
        W2 = rng.standard_normal((o, h))
        b2 = rng.standard_normal(o)
        dy = rng.standard_normal(o)

        try:
            y_sol = sol.checkpoint_forward(x, W1, b1, W2, b2)
            y_ref = _ref_forward(x, W1, b1, W2, b2)
            err_out = max(err_out, max_abs_err(y_sol, y_ref))

            dx_sol = sol.checkpoint_backward(dy, x, W1, b1, W2, b2)
            dx_ref = _ref_backward(dy, x, W1, b1, W2, b2)
            err_grad = max(err_grad, max_abs_err(dx_sol, dx_ref))
        except Exception:
            return {"output_error": float("inf"), "grad_error": float("inf")}

    return {"output_error": err_out, "grad_error": err_grad}
