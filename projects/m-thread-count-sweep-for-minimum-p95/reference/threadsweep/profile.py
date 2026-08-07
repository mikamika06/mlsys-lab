import numpy as np


def analyze_profile_gap(op_times, wall_clock):
    summed = float(np.sum(op_times))
    gap = float(wall_clock - summed)
    ratio = float(wall_clock / summed if summed > 0 else 0.0)
    return {"summed": summed, "wall_clock": wall_clock, "gap": gap, "ratio": ratio}
