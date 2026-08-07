import numpy as np


def rank_execution_providers(latencies):
    scores = {}
    for ep, times in latencies.items():
        arr = np.array(times, dtype=np.float64)
        p99 = np.percentile(arr, 99)
        scores[ep] = float(p99)
    sorted_eps = sorted(scores.keys(), key=lambda x: (scores[x], x))
    return sorted_eps, scores
