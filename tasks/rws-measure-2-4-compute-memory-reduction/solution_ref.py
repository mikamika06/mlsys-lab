import numpy as np


def measure_24_reduction(W):
    W = np.asarray(W, dtype=np.float16)
    nnz = int(np.count_nonzero(W))
    density = float(nnz / W.size)
    groups = W.size // 4
    packed_bytes = int(nnz * np.dtype(np.float16).itemsize + ((2 * groups + 7) // 8))
    return density, packed_bytes
