def broadcast_add(a: list[list[float]], b: list[float]) -> tuple[list[list[float]], callable]:
    """Forward: c = a + b (a: (n,m), b: (m,)). Returns (c, backward).

    Backward takes dc (same shape as c) and returns (da, db).
    """
    n = len(a)
    m = len(b)
    c = [[a[i][j] + b[j] for j in range(m)] for i in range(n)]

    def backward(dc: list[list[float]]) -> tuple[list[list[float]], list[float]]:
        da = dc
        db = [0.0] * m
        for j in range(m):
            col_sum = 0.0
            for i in range(n):
                col_sum += dc[i][j]
            db[j] = col_sum
        return da, db

    return c, backward
