def tiled_matmul(A: list[list[float]], B: list[list[float]], tile: int) -> list[list[float]]:
    n = len(A)
    m = len(A[0])
    p = len(B[0])
    C = [[0.0] * p for _ in range(n)]

    for ii in range(0, n, tile):
        for jj in range(0, p, tile):
            for kk in range(0, m, tile):
                i_end = min(ii + tile, n)
                j_end = min(jj + tile, p)
                k_end = min(kk + tile, m)

                for i in range(ii, i_end):
                    for k in range(kk, k_end):
                        a = A[i][k]
                        for j in range(jj, j_end):
                            C[i][j] += a * B[k][j]

    return C
