import numpy as np
from kvblock.objective import evaluate_workload_objective


def simulate_block_sweep(trace, candidate_block_sizes, total_memory_blocks, hit_penalty_weight):
    results = {}
    for b_size in candidate_block_sizes:
        seq_lengths = [req["seq_len"] for req in trace]
        shared_prefixes = list({tuple(req["prefix"]) for req in trace if "prefix" in req})
        request_prefixes = [tuple(req["prefix"]) for req in trace if "prefix" in req]

        base_obj = evaluate_workload_objective(
            seq_lengths, shared_prefixes, request_prefixes, b_size, hit_penalty_weight
        )

        total_allocated_blocks = sum((l + b_size - 1) // b_size for l in seq_lengths)
        capacity = total_memory_blocks.get(b_size, 1000000)
        overflow_penalty = max(0, total_allocated_blocks - capacity) * b_size * 2.0

        results[b_size] = float(base_obj + overflow_penalty)
    return results


def find_optimal_block_size(trace, candidate_block_sizes, total_memory_blocks, hit_penalty_weight):
    costs = simulate_block_sweep(trace, candidate_block_sizes, total_memory_blocks, hit_penalty_weight)
    sorted_sizes = sorted(candidate_block_sizes)
    best_size = min(sorted_sizes, key=lambda b: (costs[b], b))
    return best_size
