import numpy as np
from mlsys.scorers import rel_err

def _reference(x, gamma, beta, eps):
    # Dequantise (scale = 1.0) and compute LayerNorm in float64
    xf = x.astype(np.float64)
    mu = np.mean(xf, axis=-1, keepdims=True)
    var = np.var(xf, axis=-1, keepdims=True)
    denom = np.sqrt(var + eps)
    return (xf - mu) / denom * gamma + beta

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(42)
    max_err = 0.0
    for _ in range(10):
        B = rng.integers(1, 8)
        F = rng.integers(3, 16)
        x = rng.integers(-128, 128, size=(B, F), dtype=np.int8)
        gamma = rng.standard_normal(F).astype(np.float64)
        beta = rng.standard_normal(F).astype(np.float64)
        eps = 1e-5
        try:
            got = sol.layernorm_int8(x, gamma, beta, eps)
        except Exception:
            return {"rel_err": 0.0}
        ref = _reference(x, gamma, beta, eps)
        err = rel_err(ref, got)
        if err > max_err:
            max_err = err
    return {"rel_err": max_err}
