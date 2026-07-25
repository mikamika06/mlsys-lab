import numpy as np
from mlsys.scorers import max_abs_err

def _ref(x, w, eps=1e-6):
    x = np.asarray(x, dtype=np.float64)
    w = np.asarray(w, dtype=x.dtype)
    denom = np.sqrt(np.mean(x**2) + eps)
    return w * (x / denom)

def analytic_jacobian(x, w, eps=1e-6):
    n = x.size
    m = np.mean(x**2)
    denom = np.sqrt(m + eps)
    inv_denom = 1.0/denom
    inv_denom_cubed = 1.0/(denom**3)
    outer = np.outer(x, x) / n
    G = np.diag(w * inv_denom) - (w[:,None] * outer * inv_denom_cubed)
    return G

def numeric_jacobian(f, x, eps=1e-5):
    n = x.size
    y0 = f(x)
    J = np.empty((n,n), dtype=y0.dtype)
    for k in range(n):
        dx = np.zeros_like(x)
        dx[k] = eps
        y_plus = f(x + dx)
        y_minus = f(x - dx)
        J[:,k] = (y_plus - y_minus) / (2*eps)
    return J

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    max_out_err = 0.0
    max_jac_err = 0.0
    for n in [4,8]:
        x = rng.standard_normal(n).astype(np.float64)
        w = rng.uniform(-1,1,n).astype(np.float64)
        try:
            y_cand = sol.rmsnorm(x, w)
        except Exception:
            return {"max_abs_err": float("inf"), "jacobian_error": float("inf")}
        y_ref = _ref(x,w)
        out_err = max_abs_err(y_ref, y_cand)
        if out_err > max_out_err:
            max_out_err = out_err
        f = lambda v: sol.rmsnorm(v, w)
        J_num = numeric_jacobian(f, x)
        J_ana = analytic_jacobian(x,w)
        jac_err = max_abs_err(J_ana, J_num)
        if jac_err > max_jac_err:
            max_jac_err = jac_err
    return {"max_abs_err": max_out_err, "jacobian_error": max_jac_err}
