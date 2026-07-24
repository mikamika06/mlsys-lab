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
        x = rng.standard_normal((b, d))
        gamma = rng.standard_normal(d)
        beta = rng.standard_normal(d)
        try:
            out = sol.layer_norm(x, gamma, beta)
        except Exception as e:
            return {"max_abs_err": float("inf")}
        ref = _reference_layer_norm(x, gamma, beta)
        err = np.max(np.abs(out - ref))
        if err > max_err:
            max_err = err
    return {"max_abs_err": max_err}
