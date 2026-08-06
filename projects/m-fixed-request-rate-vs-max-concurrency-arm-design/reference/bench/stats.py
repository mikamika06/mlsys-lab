import numpy as np

def compute_repeats_for_claim(latencies, target_pct=0.05):
    arr = np.array(latencies, dtype=float)
    if len(arr) == 0:
        return 1
    mean_val = np.mean(arr)
    if mean_val == 0:
        return 1
    std_val = np.std(arr, ddof=1)
    cv = std_val / mean_val
    n = int(np.ceil((1.96 * cv / target_pct) ** 2))
    return max(1, min(n, 1000))
