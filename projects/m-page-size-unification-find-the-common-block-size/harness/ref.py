import math

CONFIGS = [
    {
        "layers": [
            {"index": 0, "kv_heads": 8, "head_dim": 128, "dtype_bytes": 2},
            {"index": 1, "kv_heads": 4, "head_dim": 128, "dtype_bytes": 2},
            {"index": 2, "kv_heads": 2, "head_dim": 64, "dtype_bytes": 2},
        ]
    },
    {
        "layers": [
            {"index": 0, "kv_heads": 1, "head_dim": 3, "dtype_bytes": 1},
            {"index": 1, "kv_heads": 16, "head_dim": 64, "dtype_bytes": 2},
        ]
    },
    {
        "layers": [
            {"index": 0, "kv_heads": 32, "head_dim": 128, "dtype_bytes": 2},
            {"index": 1, "kv_heads": 8, "head_dim": 128, "dtype_bytes": 2},
            {"index": 2, "kv_heads": 8, "head_dim": 128, "dtype_bytes": 2},
        ]
    },
]


def find_common_block_size(config, candidate_block_sizes=(8, 16, 32, 64), page_align_bytes=64):
    for b in candidate_block_sizes:
        valid = True
        for layer in config["layers"]:
            kv_heads = layer["kv_heads"]
            head_dim = layer["head_dim"]
            dtype_bytes = layer.get("dtype_bytes", 2)
            bytes_per_token = 2 * kv_heads * head_dim * dtype_bytes
            block_bytes = b * bytes_per_token
            if block_bytes % page_align_bytes != 0:
                valid = False
                break
        if valid:
            return b
    return candidate_block_sizes[-1]


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
