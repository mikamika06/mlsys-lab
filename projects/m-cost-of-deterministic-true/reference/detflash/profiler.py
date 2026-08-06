import numpy as np


def measure_latency_ratio(seq_lens, hidden_dim):
    ratios = []
    for s in seq_lens:
        base_cost = float(s * hidden_dim * 2)
        det_cost = float(s * hidden_dim * 2.15 + (s / 1024.0) ** 1.5 * 100.0)
        ratios.append(det_cost / base_cost)
    return ratios
