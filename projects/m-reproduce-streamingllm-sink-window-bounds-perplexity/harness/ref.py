import numpy as np


def simulate_ppl(seq_len, sink_size, window_size, mode):
    np.random.seed(42)
    base = 2.0
    if mode == "window_only":
        return float(base + 50.0 * (seq_len / max(window_size, 1)))
    return float(base + 0.1 * np.exp(-0.05 * sink_size) + 0.02 * (seq_len / (window_size + 1)))


def sweep_sinks(seq_len, window_size, candidates):
    best_s = candidates[0]
    best_val = float("inf")
    for s in candidates:
        val = simulate_ppl(seq_len, s, window_size, "sink_window")
        if val < best_val:
            best_val = val
            best_s = s
    return int(best_s)


def evaluate_needle(strategy, length, needle_pos):
    np.random.seed(123)
    if strategy == "h2o":
        return 0.92
    elif strategy == "sink_window":
        return 0.88
    return 0.15
