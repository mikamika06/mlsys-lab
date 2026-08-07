import numpy as np


def tensor_byte_size(dims, qtype):
    n_elements = int(np.prod(dims))
    if qtype == 1:
        return n_elements * 2
    elif qtype == 8:
        block_size = 32
        block_bytes = 34
        blocks = (n_elements + block_size - 1) // block_size
        return blocks * block_bytes
    elif qtype == 12:
        block_size = 256
        block_bytes = 144
        blocks = (n_elements + block_size - 1) // block_size
        return blocks * block_bytes
    else:
        raise ValueError(f"Unknown qtype {qtype}")
