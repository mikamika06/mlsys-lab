import numpy as np

def compressed_sparse_footprint(tensor):
    """Return (sparse_bytes, dense_bytes, size_ratio) for a bitmask+values
    compressed-sparse representation of an fp16 weight tensor."""
    raise NotImplementedError('your code here')
