import numpy as np

def batch_size_histogram(arrivals, batch_timeout, max_batch_size):
    """Return the histogram of formed batch sizes (length max_batch_size+1)."""
    histogram = np.zeros(max_batch_size + 1, dtype=np.int64)
    batch_start = None
    batch_size = 0
    for t in arrivals:
        if batch_size == 0:
            batch_start = t
            batch_size = 1
        else:
            batch_size += 1
        if batch_size == max_batch_size or (t - batch_start) >= batch_timeout:
            histogram[batch_size] += 1
            batch_size = 0
            batch_start = None
    if batch_size > 0:
        histogram[batch_size] += 1
    return histogram
