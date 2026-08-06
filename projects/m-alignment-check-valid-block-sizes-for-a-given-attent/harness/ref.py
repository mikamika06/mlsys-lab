import math


def validate_block_size(backend: dict, model: dict, block_size: int) -> dict:
    """Validate if a candidate block size satisfies backend and model alignment rules."""
    min_bs = backend.get("min_block_size", 1)
    max_bs = backend.get("max_block_size", 8192)
    multiple = backend.get("block_multiple", 1)
    align_bytes = backend.get("alignment_bytes", 16)

    num_heads = model["num_kv_heads"]
    head_dim = model["head_dim"]
    dtype_bytes = model["dtype_bytes"]
    is_quantized = model.get("is_quantized", False)
    group_size = model.get("group_size", 1)
    scale_bytes = model.get("scale_dtype_bytes", 4)

    if block_size < min_bs:
        return {"valid": False, "total_bytes": 0, "reason": "below_min_block_size"}
    if block_size > max_bs:
        return {"valid": False, "total_bytes": 0, "reason": "exceeds_max_block_size"}
    if block_size % multiple != 0:
        return {"valid": False, "total_bytes": 0, "reason": "tile_multiple_misaligned"}
    if is_quantized and (block_size % group_size != 0):
        return {"valid": False, "total_bytes": 0, "reason": "quant_group_misaligned"}

    payload_bytes = num_heads * 2 * block_size * head_dim * dtype_bytes
    if is_quantized:
        groups_per_block = block_size // group_size
        payload_bytes += num_heads * 2 * groups_per_block * scale_bytes

    if payload_bytes % align_bytes != 0:
        return {"valid": False, "total_bytes": payload_bytes, "reason": "memory_unaligned"}

    return {"valid": True, "total_bytes": payload_bytes, "reason": "ok"}


def filter_valid_block_sizes(backend: dict, model: dict, candidate_sizes: list) -> list:
    """Filter candidate block sizes to those passing all alignment checks."""
    valid = []
    for bs in candidate_sizes:
        res = validate_block_size(backend, model, bs)
        if res["valid"]:
            valid.append(bs)
    return valid


def select_optimal_block_size(
    backend: dict, model: dict, candidate_sizes: list, max_memory_bytes: int, prompt_lens: list
) -> dict:
    """Select the optimal block size that fits within budget and minimizes waste."""
    valid_candidates = filter_valid_block_sizes(backend, model, candidate_sizes)
    if not valid_candidates:
        return {
            "best_block_size": None,
            "allocated_bytes": 0,
            "fragmentation_bytes": 0,
            "valid_count": 0,
        }

    best_bs = None
    min_waste = None
    best_allocated = 0

    for bs in valid_candidates:
        res = validate_block_size(backend, model, bs)
        block_bytes = res["total_bytes"]

        total_blocks = sum(math.ceil(l / bs) if l > 0 else 0 for l in prompt_lens)
        alloc_bytes = total_blocks * block_bytes
        if alloc_bytes > max_memory_bytes:
            continue

        wasted_tokens = sum((math.ceil(l / bs) * bs - l) if l > 0 else 0 for l in prompt_lens)
        frag_bytes = (wasted_tokens * block_bytes) // bs

        if min_waste is None or frag_bytes < min_waste or (frag_bytes == min_waste and bs < best_bs):
            min_waste = frag_bytes
            best_bs = bs
            best_allocated = alloc_bytes

    if best_bs is None:
        return {
            "best_block_size": None,
            "allocated_bytes": 0,
            "fragmentation_bytes": 0,
            "valid_count": len(valid_candidates),
        }

    return {
        "best_block_size": best_bs,
        "allocated_bytes": best_allocated,
        "fragmentation_bytes": min_waste,
        "valid_count": len(valid_candidates),
    }


CONFIGS_M1 = [
    {
        "backend": {"min_block_size": 16, "max_block_size": 256, "block_multiple": 16, "alignment_bytes": 16},
        "model": {"num_kv_heads": 8, "head_dim": 128, "dtype_bytes": 2, "is_quantized": False},
        "candidates": [8, 16, 24, 32, 48, 64, 128, 256, 512],
    },
    {
        "backend": {"min_block_size": 16, "max_block_size": 512, "block_multiple": 16, "alignment_bytes": 32},
        "model": {
            "num_kv_heads": 16,
            "head_dim": 64,
            "dtype_bytes": 1,
            "is_quantized": True,
            "group_size": 32,
            "scale_dtype_bytes": 2,
        },
        "candidates": [16, 32, 48, 64, 96, 128, 256],
    },
    {
        "backend": {"min_block_size": 4, "max_block_size": 128, "block_multiple": 4, "alignment_bytes": 128},
        "model": {"num_kv_heads": 3, "head_dim": 64, "dtype_bytes": 2, "is_quantized": False},
        "candidates": [4, 8, 12, 16, 32, 64],
    },
    {
        "backend": {"min_block_size": 8, "max_block_size": 64, "block_multiple": 8, "alignment_bytes": 64},
        "model": {"num_kv_heads": 1, "head_dim": 33, "dtype_bytes": 2, "is_quantized": False},
        "candidates": [8, 16, 24, 32],
    },
    {
        "backend": {"min_block_size": 16, "max_block_size": 128, "block_multiple": 16, "alignment_bytes": 16},
        "model": {
            "num_kv_heads": 4,
            "head_dim": 128,
            "dtype_bytes": 1,
            "is_quantized": True,
            "group_size": 16,
            "scale_dtype_bytes": 4,
        },
        "candidates": [16, 32, 64, 128],
    },
]

CONFIGS_M2 = [
    {
        "backend": {"min_block_size": 16, "max_block_size": 256, "block_multiple": 16, "alignment_bytes": 16},
        "model": {"num_kv_heads": 8, "head_dim": 128, "dtype_bytes": 2, "is_quantized": False},
        "candidates": [16, 32, 64, 128],
        "max_memory_bytes": 10_000_000,
        "prompt_lens": [130, 250, 45, 512, 17],
    },
    {
        "backend": {"min_block_size": 16, "max_block_size": 256, "block_multiple": 16, "alignment_bytes": 16},
        "model": {"num_kv_heads": 8, "head_dim": 128, "dtype_bytes": 2, "is_quantized": False},
        "candidates": [16, 32, 64, 128],
        "max_memory_bytes": 4_100_000,
        "prompt_lens": [130, 250, 45, 512, 17],
    },
    {
        "backend": {"min_block_size": 16, "max_block_size": 256, "block_multiple": 16, "alignment_bytes": 32},
        "model": {
            "num_kv_heads": 8,
            "head_dim": 64,
            "dtype_bytes": 1,
            "is_quantized": True,
            "group_size": 32,
            "scale_dtype_bytes": 4,
        },
        "candidates": [16, 32, 64],
        "max_memory_bytes": 5_000_000,
        "prompt_lens": [100, 200, 300, 400],
    },
    {
        "backend": {"min_block_size": 16, "max_block_size": 256, "block_multiple": 16, "alignment_bytes": 16},
        "model": {"num_kv_heads": 8, "head_dim": 128, "dtype_bytes": 2, "is_quantized": False},
        "candidates": [16, 32, 64],
        "max_memory_bytes": 100,
        "prompt_lens": [100, 200],
    },
]
