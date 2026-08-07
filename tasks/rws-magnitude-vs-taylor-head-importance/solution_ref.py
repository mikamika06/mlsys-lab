import math


def _flatten(tensor):
    flat = []
    if isinstance(tensor, list):
        for sub in tensor:
            flat.extend(_flatten(sub))
    else:
        flat.append(float(tensor))
    return flat


def rank_heads_by_importance(weights: list[list[float]], grads: list[list[float]]):
    h = len(weights)
    mag_scores = []
    taylor_scores = []

    for i in range(h):
        flat_w = _flatten(weights[i])
        flat_g = _flatten(grads[i])

        sum_sq = 0.0
        sum_taylor = 0.0
        for w_val, g_val in zip(flat_w, flat_g):
            sum_sq += w_val * w_val
            sum_taylor += abs(g_val * w_val)

        mag_scores.append(math.sqrt(sum_sq))
        taylor_scores.append(sum_taylor)

    magnitude_ranking = sorted(
        range(h),
        key=lambda i: (-float(mag_scores[i]), i),
    )
    taylor_ranking = sorted(
        range(h),
        key=lambda i: (-float(taylor_scores[i]), i),
    )
    return magnitude_ranking, taylor_ranking
