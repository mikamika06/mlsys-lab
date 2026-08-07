import numpy as np
import math


def compute_required_reps(times_sample, tolerance=0.05, confidence_z=1.96):
    times = np.array(times_sample)
    mean = float(np.mean(times))
    if mean == 0:
        return 0
    std = float(np.std(times, ddof=1))
    margin = tolerance * mean
    if margin == 0:
        return 0
    return math.ceil((confidence_z * std / margin) ** 2)


def get_percentiles(times):
    times_arr = np.array(times)
    q75, q25 = np.percentile(times_arr, [75, 25])
    return float(np.median(times_arr)), float(q75 - q25)
