import numpy as np

def find_batch_size_knee(latencies: np.ndarray, throughputs: np.ndarray, slo_latency: float) -> int:
    """
    Return the index of the batch size that maximizes throughput while keeping latency <= slo_latency.
    If no such batch exists, return -1.
    """
    lat = np.asarray(latencies)
    thr = np.asarray(throughputs)
    valid = np.where(lat <= slo_latency)[0]
    if len(valid) == 0:
        return -1
    max_idx = valid[np.argmax(thr[valid])]
    return int(max_idx)
