import numpy as np

def gpu_transfer_stats(layer_sizes, window_size):
    arr = np.asarray(layer_sizes, dtype=np.int64)
    n = len(arr)
    if n == 0 or window_size <= 0:
        peak = 0
    else:
        w = min(window_size, n)
        current_sum = 0
        for i in range(w):
            current_sum += int(arr[i])
        peak = current_sum
        for i in range(1, n - w + 1):
            current_sum = current_sum - int(arr[i - 1]) + int(arr[i + w - 1])
            if current_sum > peak:
                peak = current_sum
    total = 0
    for i in range(n):
        total += int(arr[i])
    return (peak, total)
