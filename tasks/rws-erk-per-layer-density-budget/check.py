import numpy as np


def _oracle(shapes, global_density, erk_power_scale):
    n_layers = len(shapes)
    n_params = np.array([float(np.prod(s)) for s in shapes], dtype=np.float64)
    raw = np.array([(sum(s) / np.prod(s)) ** erk_power_scale for s in shapes], dtype=np.float64)
    total_params = n_params.sum()
    target_kept = global_density * total_params

    is_dense = np.zeros(n_layers, dtype=bool)
    density = np.zeros(n_layers, dtype=np.float64)

    for _ in range(n_layers + 2):
        denom = np.sum(raw[~is_dense] * n_params[~is_dense])
        kept_dense = np.sum(n_params[is_dense])
        remaining_budget = target_kept - kept_dense
        if denom <= 0:
            break
        eps = remaining_budget / denom
        cand = eps * raw
        newly_dense = (~is_dense) & (cand > 1.0)
        if not np.any(newly_dense):
            density[~is_dense] = cand[~is_dense]
            density[is_dense] = 1.0
            break
        is_dense = is_dense | newly_dense
        density[is_dense] = 1.0

    return density, n_params


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    worst_rel = 0.0
    worst_budget_rel = 0.0

    for _ in range(6):
        n_layers = int(rng.integers(3, 7))
        shapes = [
            (int(rng.integers(4, 200)), int(rng.integers(4, 200))) for _ in range(n_layers)
        ]
        global_density = float(rng.uniform(0.05, 0.5))
        power = 1.0

        exp_density, n_params = _oracle(shapes, global_density, power)

        try:
            got = np.asarray(
                sol.erk_layer_densities(list(shapes), global_density, power), dtype=np.float64
            )
        except Exception:
            worst_rel = float("inf")
            worst_budget_rel = float("inf")
            continue

        if got.shape != exp_density.shape:
            worst_rel = float("inf")
            worst_budget_rel = float("inf")
            continue

        rel = float(np.linalg.norm(got - exp_density) / (np.linalg.norm(exp_density) + 1e-12))
        worst_rel = max(worst_rel, rel)

        total_params = n_params.sum()
        achieved = float(np.sum(got * n_params) / total_params)
        budget_rel = abs(achieved - global_density) / (abs(global_density) + 1e-12)
        worst_budget_rel = max(worst_budget_rel, budget_rel)

    return {"rel_err": worst_rel, "budget_rel_err": worst_budget_rel}
