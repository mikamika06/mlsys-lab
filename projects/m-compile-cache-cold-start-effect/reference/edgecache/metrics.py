import numpy as np


def compute_population_p95(fleet_data: list) -> float:
    """
    Compute population-weighted p95 inference latency across device groups.
    """
    expanded = []
    for group in fleet_data:
        count = group["device_count"]
        latencies = group["latencies"]
        if count <= 0 or not latencies:
            continue
        for lat in latencies:
            expanded.extend([lat] * count)

    if not expanded:
        return 0.0

    return float(np.percentile(expanded, 95))
