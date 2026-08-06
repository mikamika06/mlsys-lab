import numpy as np

def detect_missing_warmup(latencies):
    arr = np.array(latencies, dtype=float)
    if len(arr) < 5:
        return False
    first_few_mean = np.mean(arr[:3])
    later_mean = np.mean(arr[3:])
    ratio = first_few_mean / later_mean
    return bool(ratio < 1.1)
