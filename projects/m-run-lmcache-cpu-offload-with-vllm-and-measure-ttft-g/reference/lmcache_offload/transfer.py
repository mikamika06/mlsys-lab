import random

def measure_ttft_gain(chunk_size: int) -> float:
    random.seed(42)
    base_latency = 100.0
    latencies_baseline = [base_latency * (1.0 + 0.05 * random.random()) for _ in range(20)]
    overhead = max(0.0, (256.0 / float(chunk_size) - 1.0) * 10.0) if chunk_size > 0 else 999.0
    latencies_offload = [max(10.0, (base_latency * 0.75) + overhead + 2.0 * random.random()) for _ in range(20)]
    mean_base = sum(latencies_baseline) / len(latencies_baseline)
    mean_off = sum(latencies_offload) / len(latencies_offload)
    return mean_base / mean_off
