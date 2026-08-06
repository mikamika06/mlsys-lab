import numpy as np

def _reference_layer_norm(x, gamma, beta):
    x = np.asarray(x, dtype=np.float64)
    gamma = np.asarray(gamma, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)
    mean = np.mean(x, axis=1, keepdims=True)
    var = np.var(x, axis=1, keepdims=True)
    eps = 1e-5
    x_hat = (x - mean) / np.sqrt(var + eps)
    return gamma * x_hat + beta

def grade(sol, fx):
    rng = np.random.default_rng(0)
    max_err = 0.0
    shapes = [(3, 4), (5, 2), (1, 6), (10, 8), (7, 1)]
    for b, d in shapes:
        x_mat = rng.standard_normal((b, d)).tolist()
        gamma_vec = rng.standard_normal(d).tolist()
        beta_vec = rng.standard_normal(d).tolist()
        try:
            out = sol.layer_norm(x_mat, gamma_vec, beta_vec)
        except Exception as e:
            return {"max_abs_err": float("inf")}
        ref = _reference_layer_norm(x_mat, gamma_vec, beta_vec)
        out_arr = np.asarray(out, dtype=np.float64)
        err = np.max(np.abs(out_arr - ref))
        if err > max_err:
            max_err = err
    return {"max_abs_err": max_err}
