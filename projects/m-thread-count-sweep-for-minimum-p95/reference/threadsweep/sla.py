import numpy as np


def compute_sla_throughput(latencies, sla_limit):
    valid = [l for l in latencies if np.percentile(l, 95) <= sla_limit]
    if not valid:
        return 0.0
    return float(1000.0 / np.mean(valid))
