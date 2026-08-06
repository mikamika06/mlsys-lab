import numpy as np


def compute_curves(profile):
    bs = np.array(profile["batch_sizes"], dtype=float)
    accept = profile["draft_accept_rate"]
    base = profile["base_latency_ms"]
    ov = profile["overhead_ms"]
    latencies = base + ov * np.sqrt(bs) / (1.0 + accept)
    throughput = bs / (latencies / 1000.0)
    return latencies, throughput
