import numpy as np


def decide_scaling_mode(activation_history, threshold=3.0):
    """Decide scaling mode."""
    history = [np.asarray(a, dtype=np.float32) for a in activation_history]
    if not history:
        return "static"
    maxs = [np.max(np.abs(a)) for a in history]
    mean_max = np.mean(maxs)
    peak_max = np.max(maxs)
    if mean_max == 0:
        return "static"
    ratio = peak_max / mean_max
    if ratio > threshold:
        return "dynamic"
    return "static"
