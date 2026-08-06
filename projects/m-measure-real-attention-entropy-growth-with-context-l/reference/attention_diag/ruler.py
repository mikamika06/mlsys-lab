import numpy as np


def reconstruct_degradation_slope(scores, context_lengths):
    x = np.log2(np.array(context_lengths, dtype=float))
    y = np.array(scores, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    ss_res = np.sum((y - predicted) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = float(1 - (ss_res / (ss_tot + 1e-12)))
    return {"slope": float(slope), "intercept": float(intercept), "r2": r2}
