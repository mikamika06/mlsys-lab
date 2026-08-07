import numpy as np
from mlsys.scorers import max_abs_err

def _f(x, y):
    """Composite function h(x,y) = log(exp(x*y))."""
    return np.log(np.exp(x * y))

def _finite_diff_vjp(f, x, y, upstream, eps=1e-5):
    n = x.size
    grad_x = np.empty_like(x)
    grad_y = np.empty_like(y)

    # Gradient w.r.t. x
    for i in range(n):
        dx = np.zeros_like(x)
        dx[i] = eps
        f_plus = f(x + dx, y)
        f_minus = f(x - dx, y)
        J_col = (f_plus - f_minus) / (2 * eps)  # shape: (n,)
        grad_x[i] = np.dot(J_col, upstream)

    # Gradient w.r.t. y
    for i in range(n):
        dy = np.zeros_like(y)
        dy[i] = eps
        f_plus = f(x, y + dy)
        f_minus = f(x, y - dy)
        J_col = (f_plus - f_minus) / (2 * eps)
        grad_y[i] = np.dot(J_col, upstream)

    return grad_x, grad_y

def grade(sol, fx) -> dict:
    # Test cases
    rng = np.random.default_rng(12345)
    for _ in range(5):
        n = rng.integers(3, 10)
        x_np = rng.standard_normal(n).astype(np.float64)
        y_np = rng.standard_normal(n).astype(np.float64)
        upstream_np = rng.standard_normal(n).astype(np.float64)

        x_list = x_np.tolist()
        y_list = y_np.tolist()
        upstream_list = upstream_np.tolist()

        try:
            got_x, got_y = sol.vjp_mul_exp_log(x_list, y_list, upstream_list)
        except Exception as e:
            return {"max_abs_err": float("inf")}

        ref_x, ref_y = _finite_diff_vjp(_f, x_np, y_np, upstream_np)

        err_x = max_abs_err(np.array(got_x), ref_x)
        err_y = max_abs_err(np.array(got_y), ref_y)
        err = max(err_x, err_y)

        if err > 1e-5:
            return {"max_abs_err": err}

    return {"max_abs_err": 0.0}
