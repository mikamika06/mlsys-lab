import numpy as np

def find_batch_size_knee(latencies: np.ndarray, throughputs: np.ndarray, slo_latency: float) -> int:
    """
    Return the index of the batch size that maximizes throughput while keeping latency <= slo_latency.
    If no such batch exists, return -1.
    """
    lat = np.asarray(latencies)
    thr = np.asarray(throughputs)
    
    best_idx = -1
    max_thr = -float('inf')
    
    for i in range(len(lat)):
        if lat[i] <= slo_latency:
            current_thr = thr[i]
            if current_thr > max_thr:
                max_thr = current_thr
                best_idx = i
                
    return int(best_idx)
