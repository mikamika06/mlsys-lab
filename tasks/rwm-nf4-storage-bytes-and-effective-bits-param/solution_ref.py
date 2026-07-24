import numpy as np


def nf4_storage(n: int, blocksize: int, scale_dtype: str) -> tuple:
    """
    NF4 storage accounting: packed 4-bit codes (2 per byte) plus one
    per-block scale of dtype `scale_dtype`.

    Returns (total_bytes: int, bits_per_param: float).
    """
    itemsize = np.dtype(scale_dtype).itemsize
    n_blocks = n // blocksize
    codes_bytes = n // 2
    scale_bytes = n_blocks * itemsize
    total_bytes = codes_bytes + scale_bytes
    bits_per_param = 8.0 * total_bytes / n
    return int(total_bytes), float(bits_per_param)
