import numpy as np
from mlsys.scorers import max_abs_err

def _ref(W, X, alpha):
    out_c = W.shape[0]
    max_W = np.max(np.abs(W.reshape(out_c, -1)), axis=1)
    max_X = np.max(np.abs(X.reshape(X.shape[0], X.shape[1], -1)), axis=(0, 2))
    return (max_X ** alpha) / (max_W ** (1 - alpha))

def grade(sol, fx):
    np.random.seed(0)
    out_c = 5
    in_c = 3
    kH, kW = 3, 3
    batch = 4
    h_out, w_out = 7, 7

    W_np = np.random.randn(out_c, in_c, kH, kW).astype(float)
    X_np = np.random.randn(batch, out_c, h_out, w_out).astype(float)

    W_list = W_np.tolist()
    X_list = X_np.tolist()

    alpha = 0.5
    try:
        got = sol.compute_migration_scales(W_list, X_list, alpha)
        ref = _ref(W_np, X_np, alpha).tolist()
    except Exception:
        return {"max_abs_err": float("inf")}

    err = max_abs_err(ref, got)
    return {"max_abs_err": err}
