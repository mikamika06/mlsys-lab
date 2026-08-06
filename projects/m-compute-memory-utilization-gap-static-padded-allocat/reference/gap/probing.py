import math


def probe_max_concurrency(total_kv_memory_bytes, seq_len_generator, block_size, bytes_per_token, safety_margin=0.05):
    effective_memory = total_kv_memory_bytes * (1.0 - safety_margin)
    bytes_per_block = block_size * bytes_per_token
    max_blocks = math.floor(effective_memory / bytes_per_block)

    used_blocks = 0
    max_seqs = 0

    for seq_len in seq_len_generator:
        needed_blocks = math.ceil(seq_len / block_size) if seq_len > 0 else 0
        if used_blocks + needed_blocks > max_blocks:
            break
        used_blocks += needed_blocks
        max_seqs += 1

    return {
        "max_concurrent_sequences": max_seqs,
        "allocated_blocks": used_blocks,
        "total_capacity_blocks": max_blocks,
        "memory_utilization": (used_blocks * bytes_per_block) / total_kv_memory_bytes if total_kv_memory_bytes > 0 else 0.0,
    }
