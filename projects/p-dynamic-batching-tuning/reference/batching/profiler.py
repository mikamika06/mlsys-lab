import numpy as np

def measure_latency_curve(batch_sizes):
    curve = {}
    for b in batch_sizes:
        base_lat = 10.0 + 2.0 * np.sqrt(b)
        curve[int(b)] = float(base_lat)
    return curve
