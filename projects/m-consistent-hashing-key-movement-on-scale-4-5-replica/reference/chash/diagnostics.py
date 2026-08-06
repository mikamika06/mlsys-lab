import numpy as np


def diagnose_hot_replicas(routing_logs, threshold_std_dev=2.0):
    """Diagnoses hot replicas from routing log lists/dicts."""
    counts = {}
    for log in routing_logs:
        r = log["replica"]
        counts[r] = counts.get(r, 0) + 1

    if not counts:
        return {"hot_replicas": [], "mean_load": 0.0, "std_dev": 0.0}

    values = list(counts.values())
    mean = float(np.mean(values))
    std = float(np.std(values))

    hot = []
    if std > 0:
        for r, count in counts.items():
            if (count - mean) / std >= threshold_std_dev:
                hot.append(r)

    return {
        "hot_replicas": sorted(hot),
        "mean_load": mean,
        "std_dev": std,
        "counts": counts
    }
