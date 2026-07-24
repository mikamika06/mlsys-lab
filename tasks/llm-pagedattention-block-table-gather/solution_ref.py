import numpy as np

def paged_gather(blocks, block_table, indices):
    n = len(indices)
    d = blocks[0][0].shape[1]
    keys = np.empty((n, d), dtype=np.float64)
    values = np.empty_like(keys)
    for i, idx in enumerate(indices):
        block_id, offset = block_table[idx]
        k_block, v_block = blocks[block_id]
        keys[i] = k_block[offset]
        values[i] = v_block[offset]
    return keys, values
