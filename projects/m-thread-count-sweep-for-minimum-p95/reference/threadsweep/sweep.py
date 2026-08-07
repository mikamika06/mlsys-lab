import numpy as np


def find_optimal_threads(threads, samples):
    p95s = [np.percentile(s, 95) for s in samples]
    best_idx = int(np.argmin(p95s))
    return threads[best_idx], best_idx
