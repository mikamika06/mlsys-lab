import numpy as np


def min_gpu_memory(layer_bytes: np.ndarray, K: int, activation_buffer: int) -> int:
    """
    Minimum GPU memory to run a layer-streamed model: the heaviest K-layer
    sliding-window sum of resident bytes, plus a fixed activation buffer.
    Uses a prefix-sum (cumsum) approach so window sums are O(1) each.
    """
    w = np.asarray(layer_bytes, dtype=np.int64).ravel()
    n = w.shape[0]
    k = int(min(max(K, 1), n))

    csum = np.concatenate(([0], np.cumsum(w)))
    window_sums = csum[k:] - csum[:-k]
    peak = int(window_sums.max())
    return peak + int(activation_buffer)
