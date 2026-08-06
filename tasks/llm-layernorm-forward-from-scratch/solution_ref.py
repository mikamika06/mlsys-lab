import math


def layer_norm(x: list[list[float]], gamma: list[float], beta: list[float]) -> list[list[float]]:
    b = len(x)
    d = len(x[0]) if b > 0 else 0
    out = [[0.0] * d for _ in range(b)]
    eps = 1e-5

    for i in range(b):
        total = 0.0
        for j in range(d):
            total += x[i][j]
        mean = total / d

        var_sum = 0.0
        for j in range(d):
            diff = x[i][j] - mean
            var_sum += diff * diff
        var = var_sum / d

        std = math.sqrt(var + eps)

        for j in range(d):
            x_hat = (x[i][j] - mean) / std
            g = gamma[j]
            b_val = beta[j]
            out[i][j] = g * x_hat + b_val

    return out
