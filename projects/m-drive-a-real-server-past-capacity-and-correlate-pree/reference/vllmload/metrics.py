import numpy as np


def compute_latency_percentile(latencies, percentile):
    if len(latencies) == 0:
        return 0.0
    return float(np.percentile(latencies, percentile))


def correlate_preemption_latency(preemptions, latencies):
    if len(preemptions) < 2 or len(latencies) < 2:
        return 0.0
    corr = np.corrcoef(preemptions, latencies)[0, 1]
    if np.isnan(corr):
        return 0.0
    return float(corr)
