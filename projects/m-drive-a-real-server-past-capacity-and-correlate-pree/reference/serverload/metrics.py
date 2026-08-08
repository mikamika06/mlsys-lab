import numpy as np

def compute_metrics(latencies, preemptions):
    p99 = np.percentile(latencies, 99)
    total_pree = np.sum(preemptions)
    return float(p99), float(total_pree)
