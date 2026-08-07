"""Concurrency curve analyzer implementation."""

import numpy as np


def locate_knee(concurrency_levels, throughputs):
    """Locate the knee point on a throughput vs concurrency curve."""
    x = np.array(concurrency_levels, dtype=float)
    y = np.array(throughputs, dtype=float)

    if len(x) < 2:
        return int(x[0])

    p1 = np.array([x[0], y[0]])
    p2 = np.array([x[-1], y[-1]])

    line_vec = p2 - p1
    line_len = np.linalg.norm(line_vec)

    if line_len == 0:
        return int(x[0])

    line_unit = line_vec / line_len

    distances = []
    for i in range(len(x)):
        p = np.array([x[i], y[i]])
        vec_p1 = p - p1
        proj = np.dot(vec_p1, line_unit) * line_unit
        perp_vec = vec_p1 - proj
        dist = np.linalg.norm(perp_vec)
        distances.append(dist)

    max_idx = int(np.argmax(distances))
    return int(x[max_idx])


def evaluate_concurrency_capacity(concurrency_levels, throughputs, target_concurrency):
    """Evaluate throughput efficiency at target concurrency against maximum throughput."""
    if target_concurrency not in concurrency_levels:
        return 0.0
    idx = concurrency_levels.index(target_concurrency)
    target_tp = throughputs[idx]
    max_tp = max(throughputs)
    return float(target_tp / max_tp) if max_tp > 0 else 0.0
