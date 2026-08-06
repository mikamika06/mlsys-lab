import math
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
    
    n_params_list = []
    for s in shapes:
        prod_s = 1.0
        for dim in s:
            prod_s *= float(dim)
        n_params_list.append(prod_s)
    n_params = np.array(n_params_list, dtype=np.float64)

    raw_list = []
    for s in shapes:
        sum_s = float(sum(s))
        prod_s = 1.0
        for dim in s:
            prod_s *= float(dim)
        raw_list.append((sum_s / prod_s) ** erk_power_scale)
    raw = np.array(raw_list, dtype=np.float64)

    total_params = 0.0
    for val in n_params_list:
        total_params += val
    target_kept = global_density * total_params

    is_dense = [False] * n_layers
    density = [0.0] * n_layers

    for _ in range(n_layers + 2):
        denom = 0.0
        for i in range(n_layers):
            if not is_dense[i]:
                denom += raw[i] * n_params[i]

        kept_dense = 0.0
        for i in range(n_layers):
            if is_dense[i]:
                kept_dense += n_params[i]

        remaining_budget = target_kept - kept_dense
        if denom <= 0:
            break

        eps = remaining_budget / denom
        
        cand = [eps * r for r in raw]
        
        newly_dense = [False] * n_layers
        any_newly_dense = False
        for i in range(n_layers):
            if (not is_dense[i]) and (cand[i] > 1.0):
                newly_dense[i] = True
                any_newly_dense = True

        if not any_newly_dense:
            for i in range(n_layers):
                if not is_dense[i]:
                    density[i] = cand[i]
                else:
                    density[i] = 1.0
            break

        for i in range(n_layers):
            is_dense[i] = is_dense[i] or newly_dense[i]
            if is_dense[i]:
                density[i] = 1.0

    return np.array(density, dtype=np.float64)
