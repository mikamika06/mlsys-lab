import numpy as np


def tiled_matmul(A, B, tile_m, tile_n, tile_k):
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)

    M, K = A.shape
    _, N = B.shape
    C = np.zeros((M, N), dtype=np.float64)

    for m0 in range(0, M, tile_m):
        for n0 in range(0, N, tile_n):
            acc = np.zeros(
                (min(tile_m, M - m0), min(tile_n, N - n0)),
                dtype=np.float64,
            )
            for k0 in range(0, K, tile_k):
                a_end = min(m0 + tile_m, M)
                b_end = min(n0 + tile_n, N)
                k_end = min(k0 + tile_k, K)

                a_tile = A[m0:a_end, k0:k_end]
                b_tile = B[k0:k_end, n0:b_end]
                acc += a_tile @ b_tile

            C[m0:a_end, n0:b_end] = acc

    return C
