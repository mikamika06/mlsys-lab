import numpy as np


def compute_scales(weights, num_samples, calibration_data):
    """Computes calibration scale based on weights and activation samples."""
    subset = calibration_data[:num_samples]
    if len(subset) == 0:
        max_val = np.max(np.abs(weights))
    else:
        activations = np.concatenate([x.flatten() for x in subset])
        act_scale = np.max(np.abs(activations)) if len(activations) > 0 else 1.0
        w_scale = np.max(np.abs(weights))
        max_val = 0.5 * w_scale + 0.5 * act_scale
    scale = max_val / 7.0 if max_val > 0 else 1.0
    return float(scale)
