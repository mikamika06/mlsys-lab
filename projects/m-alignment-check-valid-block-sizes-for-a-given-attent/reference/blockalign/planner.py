import math
from blockalign.validator import filter_valid_block_sizes, validate_block_size


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
