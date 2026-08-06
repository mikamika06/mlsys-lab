import math
import numpy as np


def _rank_desc(values):
    return sorted(range(len(values)), key=lambda i: values[i], reverse=True)


def _spearman(order_a, order_b):
    n = len(order_a)
    ra = [0.0] * n
    rb = [0.0] * n
    for i, idx in enumerate(order_a):
        ra[idx] = float(i + 1)
    for i, idx in enumerate(order_b):
        rb[idx] = float(i + 1)
    diff_sq_sum = 0.0
    for i in range(n):
        diff = ra[i] - rb[i]
        diff_sq_sum += diff * diff
    return float(1.0 - (6.0 * diff_sq_sum) / (n * (n * n - 1)))


def angular_distance_vs_block_influence(states):
    angles = []
    influences = []
    eps = 1e-12

    for layer in states:
        before = np.asarray(layer["before"], dtype=np.float64)
        after = np.asarray(layer["after"], dtype=np.float64)

        dot_val = 0.0
        norm_before_sq = 0.0
        norm_after_sq = 0.0
        diff_norm_sq = 0.0

        for b_val, a_val in zip(before, after):
            dot_val += b_val * a_val
            norm_before_sq += b_val * b_val
            norm_after_sq += a_val * a_val
            diff = a_val - b_val
            diff_norm_sq += diff * diff

        norm_before = math.sqrt(norm_before_sq)
        norm_after = math.sqrt(norm_after_sq)
        diff_norm = math.sqrt(diff_norm_sq)

        cosine = dot_val / (norm_before * norm_after + eps)
        cosine = max(-1.0, min(1.0, cosine))
        angles.append(float(math.acos(cosine)))

        influences.append(float(diff_norm / (norm_before + eps)))

    angle_order = _rank_desc(angles)
    influence_order = _rank_desc(influences)
    return angle_order, influence_order, _spearman(angle_order, influence_order)
