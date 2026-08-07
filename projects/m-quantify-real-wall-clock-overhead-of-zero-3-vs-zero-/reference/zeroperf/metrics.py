import numpy as np

def compute_overhead(z2_times, z3_times, warmup=10):
    z2_valid = np.array(z2_times[warmup:])
    z3_valid = np.array(z3_times[warmup:])
    mean_z2 = np.mean(z2_valid)
    mean_z3 = np.mean(z3_valid)
    overhead = (mean_z3 - mean_z2) / mean_z2
    return float(overhead)

def compute_rel_err(estimated, reference):
    return float(np.abs(estimated - reference) / (np.abs(reference) + 1e-8))
