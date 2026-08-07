import numpy as np

def measure_acceptance(accepted_count, total_count):
    if total_count == 0:
        return 0.0
    return float(accepted_count) / float(total_count)

def compute_speedup(baseline_time, speculative_time):
    if speculative_time <= 0:
        return 1.0
    return float(baseline_time) / float(speculative_time)
