import math


def block_sparse_attention(Q, K, V, block_mask, block_size):
    n = len(Q)
    d = len(Q[0])
    blocks = len(block_mask)

    allowed = [[False for _ in range(n)] for _ in range(n)]
    for bi in range(blocks):
        for bj in range(blocks):
            if block_mask[bi][bj]:
                r0 = bi * block_size
                c0 = bj * block_size
                for r in range(r0, min(n, r0 + block_size)):
                    for c in range(c0, min(n, c0 + block_size)):
                        allowed[r][c] = True

    scores = [[-float("inf") for _ in range(n)] for _ in range(n)]
    sqrt_d = math.sqrt(float(d))
    for i in range(n):
        for j in range(n):
            if allowed[i][j]:
                s = 0.0
                for k in range(d):
                    s += Q[i][k] * K[j][k]
                scores[i][j] = s / sqrt_d

    for i in range(n):
        max_val = -float("inf")
        for j in range(n):
            if scores[i][j] > max_val:
                max_val = scores[i][j]
        for j in range(n):
            if scores[i][j] != -float("inf"):
                scores[i][j] -= max_val

    weights = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        row_sum = 0.0
        for j in range(n):
            if scores[i][j] != -float("inf"):
                val = math.exp(scores[i][j])
                weights[i][j] = val
                row_sum += val
        if row_sum > 0.0:
            for j in range(n):
                weights[i][j] /= row_sum

    v_cols = len(V[0])
    output = [[0.0 for _ in range(v_cols)] for _ in range(n)]
    for i in range(n):
        for col in range(v_cols):
            s = 0.0
            for j in range(n):
                s += weights[i][j] * V[j][col]
            output[i][col] = s

    active = 0
    for bi in range(blocks):
        for bj in range(blocks):
            if block_mask[bi][bj]:
                active += 1

    ratio = float((n * n * d) / (active * block_size * block_size * d))
    return output, ratio
