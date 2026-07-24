import numpy as np


def tiled_matmul(A: np.ndarray, B: np.ndarray, tile: int) -> np.ndarray:
    n, m = A.shape
    _, p = B.shape
    C = np.zeros((n, p), dtype=np.float64)

    for ii in range(0, n, tile):
        for jj in range(0, p, tile):
            for kk in range(0, m, tile):
                i_end = min(ii + tile, n)
                j_end = min(jj + tile, p)
                k_end = min(kk + tile, m)

                for i in range(ii, i_end):
                    for k in range(kk, k_end):
                        a = A[i, k]
                        for j in range(jj, j_end):
                            C[i, j] += a * B[k, j]

    return C
