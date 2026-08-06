import numpy as np


def find_crossover_batch_size(batch_sizes, throughputs, bandwidths, peak_bw):
    ratios = bandwidths / peak_bw
    diffs = np.abs(ratios - 0.95 * np.max(ratios))
    idx = int(np.argmin(diffs))
    return int(batch_sizes[idx])


def diagnose_occupancy_limiter(batch_sizes, bandwidths, peak_bw):
    ratios = bandwidths / peak_bw
    return np.array([float(r) for r in ratios])
