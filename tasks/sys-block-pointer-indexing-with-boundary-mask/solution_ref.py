import numpy as np


def block_pointer_gather(A, row_start, col_start, block_m, block_n):
    out = np.zeros((block_m, block_n), dtype=A.dtype)
    rows = np.arange(row_start, row_start + block_m)
    cols = np.arange(col_start, col_start + block_n)

    row_mask = (rows >= 0) & (rows < A.shape[0])
    col_mask = (cols >= 0) & (cols < A.shape[1])

    if np.any(row_mask) and np.any(col_mask):
        out[np.ix_(row_mask, col_mask)] = A[
            np.ix_(rows[row_mask], cols[col_mask])
        ]

    return out
