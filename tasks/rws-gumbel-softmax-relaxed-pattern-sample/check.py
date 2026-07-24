import numpy as np


def _oracle(logits, g, tau):
    z = (logits + g) / tau
    z = z - np.max(z, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    worst = 0.0
    n_patterns = 6  # e.g. the C(4,2)=6 valid 2:4 sparsity patterns

    for _ in range(6):
        batch = int(rng.integers(1, 8))
        logits = rng.normal(scale=2.0, size=(batch, n_patterns))
        u = rng.uniform(1e-8, 1.0 - 1e-8, size=(batch, n_patterns))
        g = -np.log(-np.log(u))  # Gumbel(0,1) noise, FIXED (given to the solver)
        tau = float(rng.uniform(0.2, 2.0))

        exp = _oracle(logits, g, tau)

        try:
            got = np.asarray(
                sol.gumbel_softmax_relaxed(logits.copy(), g.copy(), tau), dtype=np.float64
            )
        except Exception:
            return {"max_abs_err": float("inf")}

        if got.shape != exp.shape:
            return {"max_abs_err": float("inf")}

        worst = max(worst, float(np.max(np.abs(got - exp))))

    return {"max_abs_err": worst}
