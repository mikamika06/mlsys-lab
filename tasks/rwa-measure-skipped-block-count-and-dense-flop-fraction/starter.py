import numpy as np

def measure_block_sparsity(block_mask: np.ndarray) -> tuple[int, float]:
    """Return (skipped_block_count, dense_flop_fraction) for a 2-D boolean block mask."""
    raise NotImplementedError("your code here")
