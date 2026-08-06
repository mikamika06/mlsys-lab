import numpy as np


def load_balancing_aux_loss(router_probs: np.ndarray) -> float:
    probs = np.asarray(router_probs, dtype=np.float64)
    n, e = probs.shape
    assignments = [0] * n
    for i in range(n):
        max_val = probs[i, 0]
        max_idx = 0
        for j in range(1, e):
            if probs[i, j] > max_val:
                max_val = probs[i, j]
                max_idx = j
        assignments[i] = max_idx

    counts = [0] * e
    for i in range(n):
        counts[assignments[i]] += 1

    f = [0.0] * e
    for j in range(e):
        f[j] = float(counts[j]) / float(n)

    p = [0.0] * e
    for j in range(e):
        col_sum = 0.0
        for i in range(n):
            col_sum += probs[i, j]
        p[j] = col_sum / float(n)

    total_sum = 0.0
    for j in range(e):
        total_sum += f[j] * p[j]

    return float(e * total_sum)
