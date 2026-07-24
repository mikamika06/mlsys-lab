import numpy as np

def packed_block_diagonal_mask(seq_lengths):
    total = sum(seq_lengths)
    mask = np.zeros((total, total), dtype=bool)
    start = 0
    for l in seq_lengths:
        end = start + l
        mask[start:end, start:end] = True
        start = end
    return mask
