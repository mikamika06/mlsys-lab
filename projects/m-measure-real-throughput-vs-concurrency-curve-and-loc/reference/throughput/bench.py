"""Concurrency curve benchmarking implementation."""

import math


def simulate_serving_step(concurrency, num_requests, workload_spec):
    """Simulate serving execution for given concurrency and workload spec."""
    prompt_len = workload_spec.get("prompt_len", 128)
    gen_len = workload_spec.get("gen_len", 64)
    compute_capacity = workload_spec.get("compute_capacity", 1000.0)
    memory_limit_blocks = workload_spec.get("memory_limit_blocks", 256)
    block_size = workload_spec.get("block_size", 16)

    blocks_per_req = math.ceil((prompt_len + gen_len) / block_size)
    total_blocks_needed = concurrency * blocks_per_req

    if total_blocks_needed > memory_limit_blocks:
        thrashing_penalty = 1.0 + 0.05 * ((total_blocks_needed - memory_limit_blocks) ** 1.5)
    else:
        thrashing_penalty = 1.0

    batch_overhead = 0.02 * concurrency
    eff_capacity = compute_capacity / (thrashing_penalty + batch_overhead)
    saturating_factor = 1.0 - math.exp(-0.15 * concurrency)
    actual_tps = eff_capacity * saturating_factor

    total_tokens = num_requests * gen_len
    total_time = total_tokens / max(actual_tps, 1.0)
    avg_latency = total_time * (concurrency / num_requests)

    return {
        "concurrency": concurrency,
        "throughput_tps": round(actual_tps, 4),
        "total_time_sec": round(total_time, 4),
        "avg_latency_sec": round(avg_latency, 4),
        "block_utilization": round(min(1.0, total_blocks_needed / memory_limit_blocks), 4),
    }


def measure_concurrency_curve(concurrency_levels, num_requests, workload_spec):
    """Measure throughput across a range of concurrency levels."""
    results = []
    for c in concurrency_levels:
        res = simulate_serving_step(c, num_requests, workload_spec)
        results.append(res)
    return results
