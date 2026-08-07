import math
import numpy as np


def required_runs_for_stability(latencies, target_rel_err, confidence_level=0.95):
    if len(latencies) < 2:
        return 2
    mean_val = float(np.mean(latencies))
    std_val = float(np.std(latencies, ddof=1))
    if mean_val <= 0 or std_val == 0:
        return len(latencies)
    if abs(confidence_level - 0.99) < 1e-4:
        z = 2.576
    else:
        z = 1.960
    n = ((z * std_val) / (target_rel_err * mean_val)) ** 2
    return int(math.ceil(n))
