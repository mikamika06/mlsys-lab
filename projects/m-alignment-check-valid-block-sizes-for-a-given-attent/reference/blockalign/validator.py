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
