import numpy as np


def rank_heads_by_importance(weights: np.ndarray, grads: np.ndarray):
    h = weights.shape[0]
    flat_w = weights.reshape(h, -1)
    flat_g = grads.reshape(h, -1)

    mag_scores = np.sqrt(np.sum(flat_w * flat_w, axis=1))
    taylor_scores = np.sum(np.abs(flat_g * flat_w), axis=1)

    magnitude_ranking = sorted(
        range(h),
        key=lambda i: (-float(mag_scores[i]), i),
    )
    taylor_ranking = sorted(
        range(h),
        key=lambda i: (-float(taylor_scores[i]), i),
    )
    return magnitude_ranking, taylor_ranking
