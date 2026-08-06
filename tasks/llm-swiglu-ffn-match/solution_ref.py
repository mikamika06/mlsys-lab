import math


def swiglu_ffn(
    x: list[list[float]],
    gate_w: list[list[float]],
    up_w: list[list[float]],
    down_w: list[list[float]],
) -> list[list[float]]:
    m = len(x)
    k_gate = len(x[0]) if m > 0 else 0
    n_gate = len(gate_w[0]) if len(gate_w) > 0 else 0

    gate = [[0.0] * n_gate for _ in range(m)]
    for i in range(m):
        for j in range(n_gate):
            s = 0.0
            for k in range(k_gate):
                s += x[i][k] * gate_w[k][j]
            gate[i][j] = s

    n_up = len(up_w[0]) if len(up_w) > 0 else 0
    up = [[0.0] * n_up for _ in range(m)]
    for i in range(m):
        for j in range(n_up):
            s = 0.0
            for k in range(k_gate):
                s += x[i][k] * up_w[k][j]
            up[i][j] = s

    hidden = [[0.0] * n_gate for _ in range(m)]
    for i in range(m):
        for j in range(n_gate):
            g = gate[i][j]
            silu = g / (1.0 + math.exp(-g))
            hidden[i][j] = silu * up[i][j]

    n_down = len(down_w[0]) if len(down_w) > 0 else 0
    k_down = len(down_w)
    result = [[0.0] * n_down for _ in range(m)]
    for i in range(m):
        for j in range(n_down):
            s = 0.0
            for p in range(k_down):
                s += hidden[i][p] * down_w[p][j]
            result[i][j] = s

    return result
