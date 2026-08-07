import numpy as np

def build_block_mask(seq_len, block_size, predicate):
    num_blocks = (seq_len + block_size - 1) // block_size
    block_mask = np.zeros((num_blocks, num_blocks), dtype=bool)
    for bi in range(num_blocks):
        for bj in range(num_blocks):
            q_start = bi * block_size
            q_end = min(seq_len, (bi + 1) * block_size)
            kv_start = bj * block_size
            kv_end = min(seq_len, (bj + 1) * block_size)
            q_grid, kv_grid = np.meshgrid(np.arange(q_start, q_end), np.arange(kv_start, kv_end), indexing='ij')
            res = predicate(q_grid, kv_grid)
            if np.any(res):
                block_mask[bi, bj] = True
    return block_mask
