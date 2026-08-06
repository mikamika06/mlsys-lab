import math


def streaming_attention(
    Q: list[list[float]],
    K: list[list[float]],
    V: list[list[float]],
    block_size: int,
) -> list[list[float]]:
    n = len(Q)
    d = len(Q[0])
    m = len(V[0])
    scale = 1.0 / math.sqrt(d)

    out = [[0.0] * m for _ in range(n)]
    running_max = [-math.inf] * n
    running_sum = [0.0] * n

    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        block_len = end - start

        scores = [[0.0] * block_len for _ in range(n)]
        for i in range(n):
            for j in range(block_len):
                dot_val = 0.0
                for k in range(d):
                    dot_val += Q[i][k] * K[start + j][k]
                scores[i][j] = dot_val * scale

        block_max = [0.0] * n
        for i in range(n):
            row_max = -math.inf
            for j in range(block_len):
                if scores[i][j] > row_max:
                    row_max = scores[i][j]
            block_max[i] = row_max

        new_max = [0.0] * n
        for i in range(n):
            new_max[i] = running_max[i] if running_max[i] > block_max[i] else block_max[i]

        old_scale = [0.0] * n
        for i in range(n):
            old_scale[i] = math.exp(running_max[i] - new_max[i])

        block_exp = [[0.0] * block_len for _ in range(n)]
        for i in range(n):
            for j in range(block_len):
                block_exp[i][j] = math.exp(scores[i][j] - new_max[i])

        sum_block_exp = [0.0] * n
        for i in range(n):
            s = 0.0
            for j in range(block_len):
                s += block_exp[i][j]
            sum_block_exp[i] = s

        new_sum = [0.0] * n
        for i in range(n):
            new_sum[i] = running_sum[i] * old_scale[i] + sum_block_exp[i]

        block_exp_V = [[0.0] * m for _ in range(n)]
        for i in range(n):
            for j in range(m):
                s = 0.0
                for k in range(block_len):
                    s += block_exp[i][k] * V[start + k][j]
                block_exp_V[i][j] = s

        new_out = [[0.0] * m for _ in range(n)]
        for i in range(n):
            denom = 1.0 if new_sum[i] == 0.0 else new_sum[i]
            factor1 = (running_sum[i] * old_scale[i]) / denom
            factor2 = 1.0 / denom
            for j in range(m):
                new_out[i][j] = out[i][j] * factor1 + block_exp_V[i][j] * factor2
        out = new_out

        running_max = new_max
        running_sum = new_sum

    return out
