import math
import numpy as np

def compressed_sparse_footprint(tensor):
    """Return (sparse_bytes, dense_bytes, size_ratio) for a bitmask+values
    compressed-sparse representation of an fp16 weight tensor."""
    n = len(tensor)
    nnz = int(np.count_nonzero(tensor))
    dense_bytes = int(tensor.nbytes)
    sparse_bytes = nnz * 2 + math.ceil(n / 8)
    if sparse_bytes == 0:
        ratio = float("inf")
    else:
        ratio = dense_bytes / sparse_bytes
    return sparse_bytes, dense_bytes, ratio
