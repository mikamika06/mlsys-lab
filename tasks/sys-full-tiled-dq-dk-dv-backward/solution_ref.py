import numpy as np


def flash_backward(Q, K, V, O, LSE, dO, tile_size=32):
    n, d = Q.shape
    scale = np.sqrt(float(d))

    dQ = np.zeros_like(Q, dtype=np.float64)
    dK = np.zeros_like(K, dtype=np.float64)
    dV = np.zeros_like(V, dtype=np.float64)

    D = np.sum(dO * O, axis=1)

    for row_start in range(0, n, tile_size):
        row_end = min(n, row_start + tile_size)

        q = Q[row_start:row_end]
        do = dO[row_start:row_end]
        d_row = D[row_start:row_end]

        local_dQ = np.zeros_like(q, dtype=np.float64)

        for col_start in range(0, n, tile_size):
            col_end = min(n, col_start + tile_size)

            k = K[col_start:col_end]
            v = V[col_start:col_end]

            scores = q @ k.T / scale
            p = np.exp(scores - LSE[row_start:row_end, None])

            dp = do @ v.T
            ds = p * (dp - d_row[:, None])

            local_dQ += ds @ k / scale
            dK[col_start:col_end] += ds.T @ q / scale
            dV[col_start:col_end] += p.T @ do

        dQ[row_start:row_end] = local_dQ

    return dQ, dK, dV
