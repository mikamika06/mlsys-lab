import numpy as np

def measure_block_sparsity(block_mask: np.ndarray) -> tuple[int, float]:
    """Return (skipped_block_count, dense_flop_fraction) for a 2-D boolean block mask."""
    total = block_mask.size
    if total == 0:
        return 0, 1.0
    computed = 0
    for i in range(block_mask.shape[0]):
        for j in range(block_mask.shape[1]):
            if block_mask[i, j]:
                computed += 1
    skipped = total - computed
    flop_fraction = computed / total
    return skipped, float(flop_fraction)
