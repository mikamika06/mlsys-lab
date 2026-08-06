import math


def ref_compute_utilization_gap(seq_lengths, max_seq_len, block_size, bytes_per_token):
    num_seqs = len(seq_lengths)
    static_bytes = num_seqs * max_seq_len * bytes_per_token
    total_blocks = sum(math.ceil(l / block_size) if l > 0 else 0 for l in seq_lengths)
    paged_bytes = total_blocks * block_size * bytes_per_token
    actual_bytes = sum(seq_lengths) * bytes_per_token

    return {
        "static_bytes": static_bytes,
        "paged_bytes": paged_bytes,
        "actual_tokens_bytes": actual_bytes,
        "static_waste_bytes": static_bytes - actual_bytes,
        "paged_waste_bytes": paged_bytes - actual_bytes,
        "utilization_gap_ratio": (static_bytes - paged_bytes) / static_bytes if static_bytes > 0 else 0.0,
    }


def ref_probe_max_concurrency(total_kv_memory_bytes, seq_len_generator, block_size, bytes_per_token, safety_margin=0.05):
    effective_mem = total_kv_memory_bytes * (1.0 - safety_margin)
    bytes_per_block = block_size * bytes_per_token
    max_blocks = math.floor(effective_mem / bytes_per_block)

    used_blocks = 0
    count = 0
    for l in seq_len_generator:
        b = math.ceil(l / block_size) if l > 0 else 0
        if used_blocks + b > max_blocks:
            break
        used_blocks += b
        count += 1

    return {
        "max_concurrent_sequences": count,
        "allocated_blocks": used_blocks,
        "total_capacity_blocks": max_blocks,
        "memory_utilization": (used_blocks * bytes_per_block) / total_kv_memory_bytes if total_kv_memory_bytes > 0 else 0.0,
    }
