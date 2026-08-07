def absorb_query_weight(W_Q: list[list[float]], W_UK: list[list[float]]) -> list[list[float]]:
    rows_q = len(W_Q)
    cols_q = len(W_Q[0])
    rows_uk = len(W_UK)
    cols_uk = len(W_UK[0])

    result = [[0.0 for _ in range(cols_uk)] for _ in range(cols_q)]
    for i in range(cols_q):
        for j in range(cols_uk):
            acc = 0.0
            for k in range(rows_q):
                acc += W_Q[k][i] * W_UK[k][j]
            result[i][j] = acc
    return result
