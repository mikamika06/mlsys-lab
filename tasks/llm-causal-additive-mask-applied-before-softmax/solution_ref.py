import math

def causal_masked_softmax(scores: list[list[float]]) -> list[list[float]]:
    n_rows = len(scores)
    n_cols = len(scores[0]) if n_rows > 0 else 0
    out = [[0.0] * n_cols for _ in range(n_rows)]
    for i in range(n_rows):
        row_sum = 0.0
        row_exps = []
        for j in range(n_cols):
            if j <= i:
                val = math.exp(scores[i][j])
            else:
                val = math.exp(-float('inf'))
            row_exps.append(val)
            row_sum += val
        for j in range(n_cols):
            out[i][j] = row_exps[j] / row_sum
    return out
