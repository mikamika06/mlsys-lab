def block_pointer_gather(A: list[list[float]], row_start: int, col_start: int, block_m: int, block_n: int) -> list[list[float]]:
    m = len(A)
    n = len(A[0]) if m > 0 else 0
    out = [[0.0 for _ in range(block_n)] for _ in range(block_m)]
    for i in range(block_m):
        r = row_start + i
        if 0 <= r < m:
            for j in range(block_n):
                c = col_start + j
                if 0 <= c < n:
                    out[i][j] = A[r][c]
    return out
