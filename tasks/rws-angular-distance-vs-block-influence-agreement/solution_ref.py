import numpy as np


def _rank_desc(values):
    return list(np.argsort(-np.asarray(values), kind="stable"))


def _spearman(order_a, order_b):
    ra = np.empty(len(order_a), dtype=np.float64)
    rb = np.empty(len(order_b), dtype=np.float64)
    for i, idx in enumerate(order_a):
        ra[idx] = i + 1
    for i, idx in enumerate(order_b):
        rb[idx] = i + 1
    n = len(order_a)
    return float(1.0 - 6.0 * np.sum((ra - rb) ** 2) / (n * (n * n - 1)))


def angular_distance_vs_block_influence(states):
    angles = []
    influences = []
    eps = 1e-12

    for layer in states:
        before = np.asarray(layer["before"], dtype=np.float64)
        after = np.asarray(layer["after"], dtype=np.float64)

        cosine = np.dot(before, after) / (
            np.linalg.norm(before) * np.linalg.norm(after) + eps
        )
        cosine = np.clip(cosine, -1.0, 1.0)
        angles.append(float(np.arccos(cosine)))

        influences.append(
            float(np.linalg.norm(after - before) / (np.linalg.norm(before) + eps))
        )

    angle_order = _rank_desc(angles)
    influence_order = _rank_desc(influences)
    return angle_order, influence_order, _spearman(angle_order, influence_order)
