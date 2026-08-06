import math


def _layer_norm(x: list[list[float]], gamma: list[float], beta: list[float], eps: float = 1e-5) -> list[list[float]]:
    out = []
    d = len(gamma)
    for row in x:
        mean = sum(row) / d
        var = sum((elem - mean) ** 2 for elem in row) / d
        std = math.sqrt(var + eps)
        norm_row = [((row[j] - mean) / std) * gamma[j] + beta[j] for j in range(d)]
        out.append(norm_row)
    return out


def _matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    n = len(a)
    d = len(b[0])
    m = len(b)
    out = [[0.0] * d for _ in range(n)]
    for i in range(n):
        for j in range(d):
            out[i][j] = sum(a[i][k] * b[k][j] for k in range(m))
    return out


def _add(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    n = len(a)
    d = len(a[0])
    return [[a[i][j] + b[i][j] for j in range(d)] for i in range(n)]


def transformer_block(
    x: list[list[float]],
    w_attn: list[list[float]],
    w_ff: list[list[float]],
    gamma: list[float],
    beta: list[float],
) -> list[list[float]]:
    norm_x = _layer_norm(x, gamma, beta)
    attn_out = _matmul(norm_x, w_attn)
    h1 = _add(x, attn_out)

    norm_h1 = _layer_norm(h1, gamma, beta)
    ff_out = _matmul(norm_h1, w_ff)
    return _add(h1, ff_out)
