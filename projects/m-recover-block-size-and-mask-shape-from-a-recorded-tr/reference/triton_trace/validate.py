import numpy as np


def recover_mask_shape(access_offsets, block_size, total_size):
    offsets = np.array(access_offsets, dtype=np.int64)
    active = (offsets >= 0) & (offsets < total_size)
    if len(active) == 0:
        return [False] * block_size
    mask = np.zeros(block_size, dtype=bool)
    chunk_len = min(block_size, len(active))
    mask[:chunk_len] = active[:chunk_len]
    return mask.tolist()
