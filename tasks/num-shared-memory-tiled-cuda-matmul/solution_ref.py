import numpy as np


def tiled_cuda_matmul(A, B, tile_size):
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)

    m, k = A.shape
    _, n = B.shape
    T = int(tile_size)

    C = np.zeros((m, n), dtype=np.float64)
    global_loads = 0

    for row0 in range(0, m, T):
        for col0 in range(0, n, T):
            for t0 in range(0, k, T):
                a_tile = A[row0:min(row0 + T, m), t0:min(t0 + T, k)].copy()
                b_tile = B[t0:min(t0 + T, k), col0:min(col0 + T, n)].copy()

                global_loads += a_tile.size + b_tile.size
                C[
                    row0:min(row0 + T, m),
                    col0:min(col0 + T, n),
                ] += a_tile @ b_tile

    return C, global_loads
