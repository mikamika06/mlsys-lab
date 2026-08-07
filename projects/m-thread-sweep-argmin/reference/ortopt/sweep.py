import numpy as np

def thread_sweep(latencies_dict):
    best_threads = None
    min_lat = float("inf")
    for threads, lats in latencies_dict.items():
        mean_lat = float(np.mean(lats))
        if mean_lat < min_lat:
            min_lat = mean_lat
            best_threads = threads
    return best_threads
