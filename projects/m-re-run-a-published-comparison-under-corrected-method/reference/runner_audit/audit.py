import numpy as np


def detect_context_mismatch(run_a, run_b):
    ctx_a = run_a.get("context_length")
    ctx_b = run_b.get("context_length")
    if ctx_a is None or ctx_b is None:
        return False
    return ctx_a != ctx_b


def compute_required_repeats(latencies, target_rel_error=0.05, confidence=0.95):
    arr = np.array(latencies, dtype=float)
    if len(arr) < 2:
        return int(100)
    mean_val = np.mean(arr)
    if mean_val == 0:
        return int(100)
    std_val = np.std(arr, ddof=1)
    z = 1.96
    n = ((z * std_val) / (target_rel_error * mean_val)) ** 2
    return int(np.ceil(n))
