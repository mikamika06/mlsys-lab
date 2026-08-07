import math


def emit_lse(S: list[list[float]]) -> list[float]:
    rows = len(S)
    cols = len(S[0])
    out = []
    for i in range(rows):
        m = S[i][0]
        for j in range(1, cols):
            v = S[i][j]
            if v > m:
                m = v
        total = 0.0
        for j in range(cols):
            total += math.exp(S[i][j] - m)
        out.append(m + math.log(total))
    return out
