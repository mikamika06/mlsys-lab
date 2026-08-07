import numpy as np


def measure_latency_reduction(baseline_latency, removed_heads, cost_per_head):
    total_removed = len(removed_heads)
    saved = total_removed * cost_per_head
    new_latency = max(0.0, baseline_latency - saved)
    return float(new_latency)
