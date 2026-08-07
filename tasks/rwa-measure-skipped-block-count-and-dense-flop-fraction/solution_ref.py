def measure_block_sparsity(block_mask: list[list[bool]]) -> tuple[int, float]:
    """Return (skipped_block_count, dense_flop_fraction) for a 2-D boolean block mask."""
    total = 0
    computed = 0
    for row in block_mask:
        for val in row:
            total += 1
            if val:
                computed += 1
    if total == 0:
        return 0, 1.0
    skipped = total - computed
    flop_fraction = computed / total
    return skipped, float(flop_fraction)
