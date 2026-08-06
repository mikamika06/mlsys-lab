import numpy as np


def diagnose_runs(run_a, run_b):
    rate_a = np.mean(run_a["acceptance_lengths"])
    rate_b = np.mean(run_b["acceptance_lengths"])
    if rate_a < 0.5 * rate_b:
        return "run_a_collapsed"
    if rate_b < 0.5 * rate_a:
        return "run_b_collapsed"
    return "normal"
