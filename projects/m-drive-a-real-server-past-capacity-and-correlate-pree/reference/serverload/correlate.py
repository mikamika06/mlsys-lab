import numpy as np

def correlate(latencies, preemptions):
    if len(latencies) < 2:
        return 0.0
    corr = np.corrcoef(latencies, preemptions)[0, 1]
    return float(corr)
