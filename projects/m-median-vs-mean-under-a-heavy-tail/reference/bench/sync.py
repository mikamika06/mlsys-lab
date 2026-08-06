import numpy as np

def validate_timing_agreement(event_times, wall_times, tolerance=0.1):
    ev = np.array(event_times)
    wl = np.array(wall_times)
    ratios = np.abs(ev - wl) / np.maximum(wl, 1e-8)
    return bool(np.all(ratios <= tolerance))
