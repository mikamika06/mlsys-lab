import numpy as np


def block_pointer_gather(A, row_start, col_start, block_m, block_n):
    out = np.zeros((block_m, block_n), dtype=A.dtype)
    for i in range(block_m):
        r = row_start + i
        if 0 <= r < A.shape[0]:
            for j in range(block_n):
                c = col_start + j
                if 0 <= c < A.shape[1]:
                    out[i, j] = A[r, c]
    return out
