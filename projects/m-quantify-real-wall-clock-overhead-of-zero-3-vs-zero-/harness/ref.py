import numpy as np

def generate_log_data():
    np.random.seed(42)
    z2 = 1.0 + np.random.normal(0, 0.05, 100)
    z3 = 1.25 + np.random.normal(0, 0.08, 100)
    return z2.tolist(), z3.tolist()

LOGS = generate_log_data()

def parse_log(log_str):
    lines = log_str.strip().split("\n")
    times = []
    for line in lines:
        if "step_time:" in line:
            parts = line.split("step_time:")
            try:
                times.append(float(parts[1].strip()))
            except ValueError:
                pass
    return times

def compute_overhead(z2_times, z3_times, warmup=10):
    z2_valid = np.array(z2_times[warmup:])
    z3_valid = np.array(z3_times[warmup:])
    mean_z2 = np.mean(z2_valid)
    mean_z3 = np.mean(z3_valid)
    overhead = (mean_z3 - mean_z2) / mean_z2
    return float(overhead)

def compute_rel_err(estimated, reference):
    return float(np.abs(estimated - reference) / (np.abs(reference) + 1e-8))

def summary_stats(times, warmup=10):
    arr = np.array(times[warmup:])
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95))
    }
