def occupancy_histogram(log, max_batch_size):
    hist = [0] * max_batch_size
    for sz in log:
        if 1 <= sz <= max_batch_size:
            hist[sz - 1] += 1
    return hist
