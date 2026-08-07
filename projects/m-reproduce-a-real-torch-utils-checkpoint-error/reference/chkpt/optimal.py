import numpy as np


def optimal_interval(n_layers, memory_limit):
    if n_layers <= 0:
        return 1
    val = np.sqrt(n_layers / max(1.0, memory_limit))
    opt = int(np.ceil(val))
    return max(1, min(opt, n_layers))
