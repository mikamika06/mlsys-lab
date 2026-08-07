import numpy as np

LATENCIES = {
    1: [15.2, 15.0, 15.1],
    2: [8.1, 8.0, 8.2],
    4: [10.5, 10.4, 10.6]
}

def thread_sweep(latencies_dict):
    best_threads = None
    min_lat = float("inf")
    for threads, lats in latencies_dict.items():
        mean_lat = float(np.mean(lats))
        if mean_lat < min_lat:
            min_lat = mean_lat
            best_threads = threads
    return best_threads

def run_with_iobinding(mock_session, inputs):
    outputs = []
    for inp in inputs:
        out = inp * 2.0
        outputs.append(out)
    return outputs

def compute_copy_share(total_time, copy_time):
    if total_time <= 0.0:
        return 0.0
    return float(copy_time / total_time)
