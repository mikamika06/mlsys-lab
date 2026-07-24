import numpy as np


def nf4_storage(n: int, blocksize: int, scale_dtype: str) -> tuple:
    """
    NF4 storage accounting: packed 4-bit codes (2 per byte) plus one
    per-block scale of dtype `scale_dtype`.

    Returns (total_bytes: int, bits_per_param: float). See task.md.
    """
    raise NotImplementedError('your code here')
