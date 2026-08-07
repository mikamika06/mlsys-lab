import numpy as np


def compute_migration_scales(act_max, weight_max, alpha):
    """Compute SmoothQuant per-channel migration scaling factors s."""
    act_m = np.maximum(np.asarray(act_max, dtype=np.float32), 1e-5)
    weight_m = np.maximum(np.asarray(weight_max, dtype=np.float32), 1e-5)
    scales = np.power(act_m, alpha) / np.power(weight_m, 1.0 - alpha)
    return np.maximum(scales, 1e-5)


def apply_smoothquant(activation, weight, scales):
    """Apply inverse scaling to activation and direct scaling to weights."""
    s = np.asarray(scales, dtype=np.float32)
    scaled_act = activation / s
    scaled_weight = weight * s
    return scaled_act, scaled_weight
