import numpy as np


def compute_fixed_speedup(alpha_stream, gamma):
    arr = np.asarray(alpha_stream, dtype=float)
    if len(arr) == 0:
        return 0.0
    expected_accepted = (1.0 - np.power(arr, gamma + 1)) / (1.0 - arr + 1e-12) - 1.0
    speedup = (1.0 + expected_accepted) / (1.0 + 0.1 * gamma)
    return float(np.mean(speedup))


def update_adaptive_gamma(current_gamma, recent_acceptances, min_gamma=1, max_gamma=8):
    if not recent_acceptances:
        return int(current_gamma)
    mean_acc = float(np.mean(recent_acceptances))
    if mean_acc > 0.8 and current_gamma < max_gamma:
        return int(current_gamma + 1)
    elif mean_acc < 0.5 and current_gamma > min_gamma:
        return int(current_gamma - 1)
    return int(current_gamma)
