import math
import numpy as np


def rank_heads_by_importance(weights: np.ndarray, grads: np.ndarray):
    h = weights.shape[0]
    flat_w = weights.reshape(h, -1)
    flat_g = grads.reshape(h, -1)

    d = flat_w.shape[1]
    mag_scores = np.empty(h, dtype=np.float64)
    taylor_scores = np.empty(h, dtype=np.float64)

    for i in range(h):
        sum_sq = 0.0
        sum_taylor = 0.0
        for j in range(d):
            w_val = flat_w[i, j]
            g_val = flat_g[i, j]
            sum_sq += w_val * w_val
            sum_taylor += abs(g_val * w_val)
        mag_scores[i] = math.sqrt(sum_sq)
        taylor_scores[i] = sum_taylor

    magnitude_ranking = sorted(
        range(h),
        key=lambda i: (-float(mag_scores[i]), i),
    )
    taylor_ranking = sorted(
        range(h),
        key=lambda i: (-float(taylor_scores[i]), i),
    )
    return magnitude_ranking, taylor_ranking
