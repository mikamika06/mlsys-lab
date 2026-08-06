import math

def stable_softmax(x: list[list[float]]) -> list[list[float]]:
    rows = len(x)
    cols = len(x[0]) if rows > 0 else 0
    out = [[0.0] * cols for _ in range(rows)]

    for i in range(rows):
        max_x = x[i][0]
        for j in range(1, cols):
            if x[i][j] > max_x:
                max_x = x[i][j]

        sum_exp = 0.0
        for j in range(cols):
            exp_val = math.exp(x[i][j] - max_x)
            out[i][j] = exp_val
            sum_exp += exp_val

        for j in range(cols):
            out[i][j] = out[i][j] / sum_exp

    return out
