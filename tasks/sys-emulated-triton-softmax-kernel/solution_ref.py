import math


def softmax_kernel(X: list[list[float]]) -> list[list[float]]:
    n = len(X)
    if n == 0:
        return []
    d = len(X[0])
    out = [[0.0] * d for _ in range(n)]
    for i in range(n):
        max_val = X[i][0]
        for j in range(1, d):
            if X[i][j] > max_val:
                max_val = X[i][j]
        row_sum = 0.0
        for j in range(d):
            val = math.exp(X[i][j] - max_val)
            out[i][j] = val
            row_sum += val
        for j in range(d):
            out[i][j] /= row_sum
    return out
