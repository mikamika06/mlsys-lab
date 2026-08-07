import math


def total_memory_and_waste(config, seq_len, block_size, page_align_bytes=64):
    total_allocated = 0
    total_useful = 0
    blocks_needed = math.ceil(seq_len / block_size) if seq_len > 0 else 0

    for layer in config["layers"]:
        kv_heads = layer["kv_heads"]
        head_dim = layer["head_dim"]
        dtype_bytes = layer.get("dtype_bytes", 2)
        bytes_per_token = 2 * kv_heads * head_dim * dtype_bytes

        raw_block_bytes = block_size * bytes_per_token
        aligned_block_bytes = math.ceil(raw_block_bytes / page_align_bytes) * page_align_bytes

        layer_allocated = blocks_needed * aligned_block_bytes
        layer_useful = seq_len * bytes_per_token

        total_allocated += layer_allocated
        total_useful += layer_useful

    wasted = total_allocated - total_useful
    return total_allocated, wasted
