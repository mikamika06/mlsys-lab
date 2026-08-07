import numpy as np

def measure_activation_distribution(traces):
    counts = {}
    total = 0
    for step in traces:
        for exp in step.get("activated", []):
            counts[exp] = counts.get(exp, 0) + 1
            total += 1
    if total == 0:
        return {}
    return {k: v / total for k, v in counts.items()}
