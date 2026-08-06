import numpy as np
from quant.scale import optimal_scale


def weighted_round_to_nearest(weights, imatrix, q_min, q_max, steps=15):
    weights = np.asarray(weights, dtype=np.float32)
    imatrix = np.asarray(imatrix, dtype=np.float32)
    base_scale = optimal_scale(weights, imatrix, q_min, q_max)
    scales = np.linspace(base_scale * 0.8, base_scale * 1.2, steps)
    best_scale = base_scale
    min_error = float("inf")
    best_q = None

    for s in scales:
        if s <= 0:
            continue
        q = np.clip(np.round(weights / s), q_min, q_max)
        error = np.sum(imatrix * (weights - s * q) ** 2)
        if error < min_error:
            min_error = error
            best_scale = float(s)
            best_q = q

    return best_scale, best_q
