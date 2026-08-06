import numpy as np


def severity_score(trace, transition_idx):
    window = 20
    normal_mean = np.mean(trace[:transition_idx])
    throttled_mean = np.mean(trace[transition_idx : transition_idx + window])
    ratio = throttled_mean / normal_mean
    return float(ratio)
