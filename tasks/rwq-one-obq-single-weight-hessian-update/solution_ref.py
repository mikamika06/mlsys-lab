def obq_single_weight_update(w: list[float], Hinv: list[list[float]], grid: list[float]) -> list[float]:
    n = len(w)
    m = len(grid)

    nearest = [0.0] * n
    for i in range(n):
        w_i = w[i]
        best_idx = 0
        min_diff = abs(w_i - grid[0])
        for j in range(1, m):
            diff = abs(w_i - grid[j])
            if diff < min_diff:
                min_diff = diff
                best_idx = j
        nearest[i] = grid[best_idx]

    min_cost = ((nearest[0] - w[0]) ** 2) / Hinv[0][0]
    best_k = 0
    for i in range(1, n):
        cost = ((nearest[i] - w[i]) ** 2) / Hinv[i][i]
        if cost < min_cost:
            min_cost = cost
            best_k = i

    q = nearest[best_k]
    err = q - w[best_k]
    scale = err / Hinv[best_k][best_k]

    out = [0.0] * n
    for i in range(n):
        out[i] = w[i] - scale * Hinv[i][best_k]

    out[best_k] = q
    return out
