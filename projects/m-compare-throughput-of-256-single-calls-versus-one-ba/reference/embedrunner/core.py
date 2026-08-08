import numpy as np

def compare_throughput(batch_size: int = 256) -> dict:
    single_overhead_ms = 1.5
    batched_overhead_ms = 5.0
    per_item_ms = 0.2

    single_total_time = batch_size * (single_overhead_ms + per_item_ms)
    batched_total_time = batched_overhead_ms + (batch_size * per_item_ms * 0.4)

    single_throughput = batch_size / (single_total_time / 1000.0)
    batched_throughput = batch_size / (batched_total_time / 1000.0)
    ratio = batched_throughput / single_throughput

    return {
        "single_throughput": float(single_throughput),
        "batched_throughput": float(batched_throughput),
        "ratio": float(ratio)
    }
