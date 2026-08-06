import numpy as np

def reconstruct_batch_sizes(events, time_grid):
    counts = np.zeros_like(time_grid, dtype=int)
    for start, end in events:
        mask = (time_grid >= start) & (time_grid < end)
        counts += mask.astype(int)
    return counts
