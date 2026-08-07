def sgmv(X: list[list[float]], adapters: list[list[list[float]]], segments: list[tuple[int, int, int]]) -> list[list[float]]:
    n = len(X)
    d = len(X[0])
    m = len(adapters[0][0])
    out = [[0.0] * m for _ in range(n)]
    for start, end, adapter_id in segments:
        W = adapters[adapter_id]
        for i in range(start, end):
            row = X[i]
            for j in range(m):
                val = 0.0
                for k in range(d):
                    val += row[k] * W[k][j]
                out[i][j] = val
    return out
