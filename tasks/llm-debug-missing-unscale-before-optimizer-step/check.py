import numpy as np

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(42)
    total_rel = 0.0
    n_trials = 10
    for _ in range(n_trials):
        n_params = rng.integers(2, 6)
        params = [rng.normal(size=rng.integers(4, 16)).astype(np.float64) for _ in range(n_params)]
        true_grads = [rng.normal(size=p.shape).astype(np.float64) for p in params]
        scale = float(2 ** rng.integers(8, 16))
        lr = float(rng.uniform(1e-4, 1e-2))

        # Scaled gradients (what the buggy code receives)
        scaled_grads = [g * scale for g in true_grads]

        # Reference: unscale then apply
        ref_params = [p - lr * (g / scale) for p, g in zip(params, scaled_grads)]

        try:
            got_params = sol.optimizer_step(
                [p.copy() for p in params],
                [g.copy() for g in scaled_grads],
                scale, lr
            )
        except Exception:
            return {"rel_err": float("inf")}

        err = 0.0
        norm = 0.0
        for ref, got in zip(ref_params, got_params):
            diff = np.asarray(got, dtype=np.float64) - ref
            err += float(np.sum(diff ** 2))
            norm += float(np.sum(ref ** 2))
        total_rel += (err / (norm + 1e-12)) ** 0.5

    return {"rel_err": total_rel / n_trials}
