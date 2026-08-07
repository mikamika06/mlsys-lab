def obs_update(W: list[list[float]], Hinv: list[list[float]], q: int) -> list[list[float]]:
    m = len(W)
    n = len(W[0])
    out = [[W[i][j] for j in range(n)] for i in range(m)]
    scale = Hinv[q][q]
    for j in range(n):
        if j != q:
            factor = Hinv[q][j] / scale
            for i in range(m):
                out[i][j] -= out[i][q] * factor
    for i in range(m):
        out[i][q] = 0.0
    return out
