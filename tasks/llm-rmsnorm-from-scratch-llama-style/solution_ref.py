import math

def rmsnorm(x: list[float], weight: list[float], eps: float = 1e-6) -> list[float]:
    n = len(x)

    sq_sum = 0.0
    for i in range(n):
        val = float(x[i])
        sq_sum += val * val

    mean_sq = sq_sum / n
    denom = math.sqrt(mean_sq + eps)

    out = []
    for i in range(n):
        out.append(float(weight[i]) * (float(x[i]) / denom))

    return out
