"""Oracle reference implementation for test harness."""

import math
import numpy as np

CONCURRENCY_LEVELS = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128]
NUM_REQUESTS = 100
WORKLOAD_SPEC = {
    "prompt_len": 128,
    "gen_len": 64,
    "compute_capacity": 1200.0,
    "memory_limit_blocks": 256,
    "block_size": 16,
}


def simulate_serving_step(concurrency, num_requests, workload_spec):
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
    return [simulate_serving_step(c, num_requests, workload_spec) for c in concurrency_levels]


def locate_knee(concurrency_levels, throughputs):
    x = np.array(concurrency_levels, dtype=float)
    y = np.array(throughputs, dtype=float)

    if len(x) < 2:
        return int(x[0])

    p1 = np.array([x[0], y[0]])
    p2 = np.array([x[-1], y[-1]])

    line_vec = p2 - p1
    line_len = np.linalg.norm(line_vec)

    if line_len == 0:
        return int(x[0])

    line_unit = line_vec / line_len

    distances = []
    for i in range(len(x)):
        p = np.array([x[i], y[i]])
        vec_p1 = p - p1
        proj = np.dot(vec_p1, line_unit) * line_unit
        perp_vec = vec_p1 - proj
        dist = np.linalg.norm(perp_vec)
        distances.append(dist)

    max_idx = int(np.argmax(distances))
    return int(x[max_idx])


def evaluate_concurrency_capacity(concurrency_levels, throughputs, target_concurrency):
    if target_concurrency not in concurrency_levels:
        return 0.0
    idx = concurrency_levels.index(target_concurrency)
    target_tp = throughputs[idx]
    max_tp = max(throughputs)
    return float(target_tp / max_tp) if max_tp > 0 else 0.0
