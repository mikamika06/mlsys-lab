import numpy as np


def detect_transition(trace):
    n = len(trace)
    best_idx = 0
    max_diff = 0.0
    window = 20
    for i in range(window, n - window):
        left_mean = np.mean(trace[i - window : i])
        right_mean = np.mean(trace[i : i + window])
        diff = right_mean - left_mean
        if diff > max_diff:
            max_diff = diff
            best_idx = i
    return best_idx
