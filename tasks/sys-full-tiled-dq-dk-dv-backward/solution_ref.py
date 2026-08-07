import math


def flash_backward(
    Q: list[list[float]],
    K: list[list[float]],
    V: list[list[float]],
    O: list[list[float]],
    LSE: list[float],
    dO: list[list[float]],
    tile_size: int = 32,
) -> tuple[list[list[float]], list[list[float]], list[list[float]]]:
    n = len(Q)
    d = len(Q[0])
    dv = len(V[0])
    scale = math.sqrt(float(d))

    dQ = [[0.0] * d for _ in range(n)]
    dK = [[0.0] * d for _ in range(n)]
    dV = [[0.0] * dv for _ in range(n)]

    D = [0.0] * n
    for i in range(n):
        s_val = 0.0
        for j in range(dv):
            s_val += dO[i][j] * O[i][j]
        D[i] = s_val

    for row_start in range(0, n, tile_size):
        row_end = min(n, row_start + tile_size)
        len_row = row_end - row_start

        q = [Q[i] for i in range(row_start, row_end)]
        do = [dO[i] for i in range(row_start, row_end)]

        local_dQ = [[0.0] * d for _ in range(len_row)]

        for col_start in range(0, n, tile_size):
            col_end = min(n, col_start + tile_size)
            len_col = col_end - col_start

            k = [K[j] for j in range(col_start, col_end)]
            v = [V[j] for j in range(col_start, col_end)]

            scores = [[0.0] * len_col for _ in range(len_row)]
            for i in range(len_row):
                for j in range(len_col):
                    dot = 0.0
                    for k_idx in range(d):
                        dot += q[i][k_idx] * k[j][k_idx]
                    scores[i][j] = dot / scale

            p = [[0.0] * len_col for _ in range(len_row)]
            for i in range(len_row):
                lse_val = LSE[row_start + i]
                for j in range(len_col):
                    p[i][j] = math.exp(scores[i][j] - lse_val)

            dp = [[0.0] * len_col for _ in range(len_row)]
            for i in range(len_row):
                for j in range(len_col):
                    dot = 0.0
                    for k_idx in range(dv):
                        dot += do[i][k_idx] * v[j][k_idx]
                    dp[i][j] = dot

            ds = [[0.0] * len_col for _ in range(len_row)]
            for i in range(len_row):
                d_val = D[row_start + i]
                for j in range(len_col):
                    ds[i][j] = p[i][j] * (dp[i][j] - d_val)

            ds_k = [[0.0] * d for _ in range(len_row)]
            for i in range(len_row):
                for j in range(d):
                    dot = 0.0
                    for k_idx in range(len_col):
                        dot += ds[i][k_idx] * k[k_idx][j]
                    ds_k[i][j] = dot / scale

            for i in range(len_row):
                for j in range(d):
                    local_dQ[i][j] += ds_k[i][j]

            ds_q = [[0.0] * d for _ in range(len_col)]
            for i in range(len_col):
                for j in range(d):
                    dot = 0.0
                    for k_idx in range(len_row):
                        dot += ds[k_idx][i] * q[k_idx][j]
                    ds_q[i][j] = dot / scale

            for i in range(len_col):
                for j in range(d):
                    dK[col_start + i][j] += ds_q[i][j]

            p_do = [[0.0] * dv for _ in range(len_col)]
            for i in range(len_col):
                for j in range(dv):
                    dot = 0.0
                    for k_idx in range(len_row):
                        dot += p[k_idx][i] * do[k_idx][j]
                    p_do[i][j] = dot

            for i in range(len_col):
                for j in range(dv):
                    dV[col_start + i][j] += p_do[i][j]

        for i in range(len_row):
            for j in range(d):
                dQ[row_start + i][j] = local_dQ[i][j]

    return dQ, dK, dV
