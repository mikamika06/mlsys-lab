import numpy as np


def measure_trials_curve(trials):
    latencies = []
    for t in trials:
        lat = 10.0 * np.exp(-float(t) / 300.0) + 6.5
        latencies.append(float(lat))
    return np.array(trials, dtype=np.int64), np.array(latencies, dtype=np.float64)
