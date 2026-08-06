import math


def apply_rope(x: list[float], position: int) -> list[float]:
    d = len(x)
    half = d // 2

    out = [0.0] * d
    for i in range(half):
        theta = position * (10000.0 ** (-2.0 * i / d))
        c = math.cos(theta)
        s = math.sin(theta)
        a = x[i]
        b = x[i + half]
        out[i] = a * c - b * s
        out[i + half] = a * s + b * c
    return out
