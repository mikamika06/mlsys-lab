import numpy as np


def erk_layer_densities(shapes, global_density: float, erk_power_scale: float = 1.0):
    """
    Erdos-Renyi-Kernel (ERK) per-layer density allocation: raw density is
    proportional to (sum(shape) / prod(shape)) ** erk_power_scale (bigger
    for layers with fewer parameters relative to fan-in+fan-out), scaled
    by a single epsilon so the parameter-weighted average density equals
    `global_density`. Layers whose raw*epsilon would exceed 1.0 are
    pinned dense (density=1) and epsilon is iteratively recomputed over
    the remaining layers.
    """
    shapes = [tuple(s) for s in shapes]
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

    return density
