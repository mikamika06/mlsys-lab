def kv_memory_usage(T: int, d: int, nbits: int = 4,
                    R: int = 0, group_size: int = 1) -> int:
    """
    Compute the total number of bytes required to store a compressed KV table.

    Parameters
    ----------
    T : int
        Total number of tokens in the cache.
    d : int
        Dimensionality of each value vector.
    nbits : int, optional
        Bits per quantized element (default 4).
    R : int, optional
        Size of the residual window that stays in fp16 (default 0).
    group_size : int, optional
        Number of tokens per scale/zero‑point group (default 1).

    Returns
    -------
    int
        Total number of bytes needed for the compressed KV table.
    """
    if T <= 0 or d <= 0:
        raise ValueError("T and d must be positive integers")
    if nbits not in {4, 8}:
        raise ValueError("nbits currently supported: 4 or 8")
    if R < 0 or R > T:
        raise ValueError("R must satisfy 0 <= R <= T")
    if group_size <= 0:
        raise ValueError("group_size must be positive")

    # int‑quantized part (except residual window)
    quant_int_bytes = (nbits / 8.0) * (T - R) * d

    # residual window in fp16
    residual_fp16_bytes = 2.0 * R * d

    # number of groups for scales and zero‑points
    remaining = max(0, T - R)
    if remaining == 0:
        group_count = 0
    else:
        group_count = (remaining + group_size - 1) // group_size

    # each group stores a fp16 scale (2 bytes) and two packed int4 zero‑points (1 byte)
    per_group_overhead = 3  # bytes

    overhead_bytes = group_count * per_group_overhead

    total_bytes = quant_int_bytes + residual_fp16_bytes + overhead_bytes
    return int(round(total_bytes))
