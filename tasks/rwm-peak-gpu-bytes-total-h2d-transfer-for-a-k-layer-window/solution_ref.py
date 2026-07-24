import numpy as np

def gpu_transfer_stats(layer_sizes, window_size):
    arr = np.asarray(layer_sizes, dtype=np.int64)
    n = len(arr)
    if n == 0 or window_size <= 0:
        peak = 0
    else:
        w = min(window_size, n)
        cumsum = np.concatenate([[0], np.cumsum(arr)])
        sums = cumsum[w:] - cumsum[:-w]
        peak = int(np.max(sums)) if sums.size > 0 else int(np.sum(arr))
    total = int(np.sum(arr))
    return (peak, total)
