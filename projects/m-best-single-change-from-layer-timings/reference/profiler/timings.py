import numpy as np


def best_single_change(layer_times):
    if not layer_times:
        return -1
    times = np.array([t for _, t in layer_times], dtype=float)
    idx = int(np.argmin(times))
    return int(layer_times[idx][0])
