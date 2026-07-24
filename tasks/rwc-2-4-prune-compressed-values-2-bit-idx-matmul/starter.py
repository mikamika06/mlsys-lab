import numpy as np


def prune24_compress_and_matmul(W: np.ndarray, X: np.ndarray):
    """
    2:4 structured sparsity: prune every group of 4 consecutive columns in
    each row of W down to its 2 largest-magnitude values. Build the
    compressed (values, indices) representation, reconstruct the pruned
    matrix from it, and multiply by X.

    Returns (mask, values, indices, output). See task.md.
    """
    raise NotImplementedError('your code here')
