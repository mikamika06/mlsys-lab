import numpy as np


def safe_vector_add(x, y, n_elements, block_size=64):
    out = np.zeros(n_elements, dtype=x.dtype)
    num_blocks = (n_elements + block_size - 1) // block_size
    for pid in range(num_blocks):
        offsets = pid * block_size + np.arange(block_size)
        mask = offsets < n_elements
        x_tok = np.where(mask, x[np.minimum(offsets, n_elements - 1)], 0.0)
        y_tok = np.where(mask, y[np.minimum(offsets, n_elements - 1)], 0.0)
        res = x_tok + y_tok
        valid_offsets = offsets[mask]
        out[valid_offsets] = res[mask]
    return out


def catch_unmasked_store(x, n_elements, block_size=64):
    out = np.zeros(n_elements, dtype=x.dtype)
    num_blocks = (n_elements + block_size - 1) // block_size
    try:
        for pid in range(num_blocks):
            offsets = pid * block_size + np.arange(block_size)
            mask = offsets < n_elements
            x_tok = np.where(mask, x[np.minimum(offsets, n_elements - 1)], 0.0)
            out[offsets] = x_tok
    except Exception as e:
        return True, str(e)
    return False, ""
