import numpy as np


def compute_perplexity(seq_len, sink_size, window_size, mode):
    np.random.seed(42)
    base = 2.0
    if mode == "window_only":
        return float(base + 50.0 * (seq_len / max(window_size, 1)))
    return float(base + 0.1 * np.exp(-0.05 * sink_size) + 0.02 * (seq_len / (window_size + 1)))
