import math


def allocate_pages(config, seq_len, block_size, page_align_bytes=64):
    allocations = {}
    blocks_needed = math.ceil(seq_len / block_size) if seq_len > 0 else 0
    for idx, layer in enumerate(config["layers"]):
        kv_heads = layer["kv_heads"]
        head_dim = layer["head_dim"]
        dtype_bytes = layer.get("dtype_bytes", 2)
        bytes_per_token = 2 * kv_heads * head_dim * dtype_bytes
        raw_block_bytes = block_size * bytes_per_token
        aligned_block_bytes = math.ceil(raw_block_bytes / page_align_bytes) * page_align_bytes
        allocations[idx] = blocks_needed * aligned_block_bytes
    return allocations
