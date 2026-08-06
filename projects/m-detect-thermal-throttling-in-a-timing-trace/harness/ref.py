import numpy as np


def generate_traces():
    rng = np.random.default_rng(42)
    traces = []
    for _ in range(5):
        n = 1000
        base = rng.normal(100.0, 2.0, n)
        transition = rng.integers(400, 600)
        drop_factor = rng.uniform(1.25, 1.5)
        base[transition:] *= drop_factor
        base[transition:] += rng.normal(0.0, 3.0, n - transition)
        traces.append((base, transition, drop_factor))
    return traces


TRACES = generate_traces()


def find_transitions(trace):
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


def compute_severity(trace, transition_idx):
    window = 20
    normal_mean = np.mean(trace[:transition_idx])
    throttled_mean = np.mean(trace[transition_idx : transition_idx + window])
    ratio = throttled_mean / normal_mean
    return float(ratio)
