def measure_latency(base_latency, removed_count, total_heads):
    fraction = removed_count / total_heads
    return float(base_latency * (1.0 - 0.5 * fraction))
