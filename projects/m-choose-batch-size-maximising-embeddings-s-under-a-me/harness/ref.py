import numpy as np


def compute_optimal_batch_size(candidates, profile_fn, memory_cap):
    best_bs = None
    max_throughput = -1.0
    for bs in candidate_batch_sizes(candidates):
        throughput, peak_mem = profile_fn(bs)
        if peak_mem <= memory_cap and throughput > max_throughput:
            max_throughput = throughput
            best_bs = bs
    return best_bs, max_throughput


def candidate_batch_sizes(candidates):
    return sorted(list(candidates))


def process_sequence_truncation(tokens, num_ctx, policy):
    tokens = list(tokens)
    length = len(tokens)
    if length <= num_ctx:
        return {
            "tokens": tokens,
            "truncated": False,
            "original_length": length,
            "final_length": length,
            "policy_applied": policy
        }

    if policy == "error":
        raise ValueError(f"Sequence length {length} exceeds num_ctx {num_ctx}")
    elif policy == "truncate_right":
        truncated_tokens = tokens[:num_ctx]
    elif policy == "truncate_left":
        truncated_tokens = tokens[-num_ctx:]
    else:
        raise ValueError(f"Unknown truncation policy: {policy}")

    return {
        "tokens": truncated_tokens,
        "truncated": True,
        "original_length": length,
        "final_length": num_ctx,
        "policy_applied": policy
    }


def generate_benchmark_data(seed=42):
    rng = np.random.RandomState(seed)
    candidates = [1, 2, 4, 8, 16, 32, 64, 128]
    memory_cap = 4096.0

    def mock_profile(bs):
        mem = 128.0 + bs * 50.0 + (bs ** 1.5) * 0.5
        time_per_emb = 0.01 + 0.05 / (bs ** 0.3)
        throughput = bs / (bs * time_per_emb) if bs > 0 else 0.0
        return float(throughput), float(mem)

    return candidates, mock_profile, memory_cap
