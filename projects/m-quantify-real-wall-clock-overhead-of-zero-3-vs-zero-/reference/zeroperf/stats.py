import numpy as np

def summary_stats(times, warmup=10):
    arr = np.array(times[warmup:])
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95))
    }
