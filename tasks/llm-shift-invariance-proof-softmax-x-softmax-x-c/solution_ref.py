import math


def softmax(x: list[float]) -> list[float]:
    """Numerically stable softmax computation."""
    n = len(x)
    if n == 0:
        return []

    max_val = float(x[0])
    for i in range(1, n):
        val = float(x[i])
        if val > max_val:
            max_val = val

    exp_vals = [0.0] * n
    sum_e = 0.0
    for i in range(n):
        ev = math.exp(float(x[i]) - max_val)
        exp_vals[i] = ev
        sum_e += ev

    out = [0.0] * n
    for i in range(n):
        out[i] = exp_vals[i] / sum_e

    return out
