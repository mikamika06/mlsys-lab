import numpy as np


def masked_block_sum(x: np.ndarray, block_size: int) -> np.ndarray:
    # TODO: missing boundary mask.
    # This simulates an unmasked block load by treating out-of-range lanes
    # as if they contained nonzero memory values.
    x = np.asarray(x)
    n = len(x)
    blocks = (n + block_size - 1) // block_size
    out = np.zeros(blocks, dtype=np.float64)

    for k in range(blocks):
        start = k * block_size
        total = 0.0
        for lane in range(block_size):
            idx = start + lane
            if idx < n:
                total += float(x[idx])
            else:
                total += 1.0
        out[k] = total

    return out
