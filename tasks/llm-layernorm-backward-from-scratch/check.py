import numpy as np
from mlsys.scorers import rel_err

def _forward(x, gamma, beta, eps):
    mu = x.mean(axis=1, keepdims=True)
    var = x.var(axis=1, keepdims=True) + eps
    std_inv = 1.0 / np.sqrt(var)
    return gamma * (x - mu) * std_inv + beta

def _loss(x, dy, gamma, beta, eps):
    y = _forward(x, gamma, beta, eps)
    return np.sum(dy * y)

def numeric_grad(x, dy, gamma, beta, eps=1e-5, h=1e-6):
    grad = np.empty_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        orig = x[idx]
        x[idx] = orig + h
        f_plus = _loss(x, dy, gamma, beta, eps)
        x[idx] = orig - h
        f_minus = _loss(x, dy, gamma, beta, eps)
        grad[idx] = (f_plus - f_minus) / (2 * h)
        x[idx] = orig
        it.iternext()
    return grad

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    N, D = 4, 5
    x      = rng.standard_normal((N, D))
    gamma  = rng.standard_normal(D)
    beta   = rng.standard_normal(D)
    dy     = rng.standard_normal((N, D))

    try:
        dx_ref = numeric_grad(x.copy(), dy, gamma, beta)
        dx_sol = sol.compute_dx(dy, x, gamma, beta)
    except Exception as e:
        return {"rel_err": 0.0}

    err = rel_err(dx_ref, dx_sol)
    return {"rel_err": err}
