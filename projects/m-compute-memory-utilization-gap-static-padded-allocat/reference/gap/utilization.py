import math


def calculate_static_memory(seq_lengths, max_seq_len, bytes_per_token):
    num_seqs = len(seq_lengths)
    return num_seqs * max_seq_len * bytes_per_token


def calculate_paged_memory(seq_lengths, block_size, bytes_per_token):
    total_blocks = 0
    for length in seq_lengths:
        blocks = math.ceil(length / block_size) if length > 0 else 0
        total_blocks += blocks
    return total_blocks * block_size * bytes_per_token


def compute_utilization_gap(seq_lengths, max_seq_len, block_size, bytes_per_token):
    if not seq_lengths:
        return {
            "static_bytes": 0,
            "paged_bytes": 0,
            "actual_tokens_bytes": 0,
            "static_waste_bytes": 0,
            "paged_waste_bytes": 0,
            "utilization_gap_ratio": 0.0,
        }

    static_bytes = calculate_static_memory(seq_lengths, max_seq_len, bytes_per_token)
    paged_bytes = calculate_paged_memory(seq_lengths, block_size, bytes_per_token)
    actual_tokens = sum(seq_lengths)
    actual_tokens_bytes = actual_tokens * bytes_per_token

    static_waste = static_bytes - actual_tokens_bytes
    paged_waste = paged_bytes - actual_tokens_bytes

    gap_ratio = (static_bytes - paged_bytes) / static_bytes if static_bytes > 0 else 0.0

    return {
        "static_bytes": static_bytes,
        "paged_bytes": paged_bytes,
        "actual_tokens_bytes": actual_tokens_bytes,
        "static_waste_bytes": static_waste,
        "paged_waste_bytes": paged_waste,
        "utilization_gap_ratio": gap_ratio,
    }
