import numpy as np

def check_littles_law(arrivals, queue_lengths, latencies):
    expected_l = np.array(arrivals) * np.array(latencies)
    actual_l = np.array(queue_lengths)
    diff = np.abs(expected_l - actual_l)
    return float(np.mean(diff))
