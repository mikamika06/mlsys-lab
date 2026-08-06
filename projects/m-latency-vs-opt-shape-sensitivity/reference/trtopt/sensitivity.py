"""Profile sensitivity analysis."""

import numpy as np

def compute_shape_sensitivity(profile, cost_fn):
    """Compute sensitivity score across dynamic range."""
    min_shape = np.array(profile["min"], dtype=np.float64)
    opt_shape = np.array(profile["opt"], dtype=np.float64)
    max_shape = np.array(profile["max"], dtype=np.float64)

    cost_min = cost_fn(min_shape, opt_shape)
    cost_opt = cost_fn(opt_shape, opt_shape)
    cost_max = cost_fn(max_shape, opt_shape)

    sens_min = float(abs(cost_min - cost_opt) / max(cost_opt, 1e-6))
    sens_max = float(abs(cost_max - cost_opt) / max(cost_opt, 1e-6))
    total_sensitivity = float(sens_min + sens_max)

    return {
        "sens_min": sens_min,
        "sens_max": sens_max,
        "total_sensitivity": total_sensitivity,
    }
