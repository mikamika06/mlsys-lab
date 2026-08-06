import numpy as np


def nf4_block_absmax_scales(W: np.ndarray) -> np.ndarray:
    w = np.asarray(W, dtype=np.float64).reshape(-1)
    blocks = w.reshape(-1, 64)
    num_blocks = blocks.shape[0]
    scales = np.empty(num_blocks, dtype=np.float64)
    for i in range(num_blocks):
        max_val = abs(blocks[i, 0])
        for j in range(1, 64):
            val = abs(blocks[i, j])
            if val > max_val:
                max_val = val
        scales[i] = max_val
    return scales
