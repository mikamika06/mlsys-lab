import numpy as np


def analyze_latencies(stateless_times, stateful_times):
    stateless_arr = np.array(stateless_times, dtype=np.float64)
    stateful_arr = np.array(stateful_times, dtype=np.float64)
    ratio = np.mean(stateful_arr) / (np.mean(stateless_arr) + 1e-9)
    return {
        "stateless_mean": float(np.mean(stateless_arr)),
        "stateful_mean": float(np.mean(stateful_arr)),
        "ratio": float(ratio),
        "valid": bool(np.all(stateless_arr > 0) and np.all(stateful_arr > 0))
    }
