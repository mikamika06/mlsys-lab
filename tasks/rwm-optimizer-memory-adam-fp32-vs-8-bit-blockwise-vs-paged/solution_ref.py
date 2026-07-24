def estimate_memory(num_params: int, block_size: int) -> tuple[int, int, int]:
    """
    Return the memory cost in bytes for three Adam optimizer representations:
    FP32, 8‑bit blockwise with per‑block maxima, and paged 8‑bit without steady overhead.
    """
    if num_params < 0 or block_size <= 0:
        raise ValueError("num_params must be non‑negative and block_size positive")

    fp32_bytes = 8 * num_params
    blocks = (num_params + block_size - 1) // block_size
    blockwise_bytes = 2 * num_params + 2 * blocks * 4
    paged_bytes = 2 * num_params

    return fp32_bytes, blockwise_bytes, paged_bytes
