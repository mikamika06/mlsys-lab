import numpy as np


def _gelu_tanh(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def _oracle(X, bias, residual, scale):
    """Op-by-op reference: each stage materialized separately, exactly as a
    naive (unfused) eager-mode implementation would run it."""
    h = X + bias
    h = _gelu_tanh(h)
    h = h + residual
    h = h * scale
    return h


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(4)
    worst = 0.0

    for _ in range(6):
        batch = int(rng.integers(2, 8))
        dim = int(rng.integers(3, 16))
        X = rng.standard_normal((batch, dim)) * 3.0
        bias = rng.standard_normal(dim)
        residual = rng.standard_normal((batch, dim))
        scale = float(rng.uniform(0.5, 2.0))

        ref = _oracle(X, bias, residual, scale)
        try:
            got = np.asarray(sol.fused_elementwise_chain(X.copy(), bias.copy(), residual.copy(), scale), dtype=np.float64)
        except Exception:
            return {"max_abs_err": float("inf")}

        if got.shape != ref.shape:
            return {"max_abs_err": float("inf")}

        worst = max(worst, float(np.max(np.abs(got - ref))))

    return {"max_abs_err": worst}
