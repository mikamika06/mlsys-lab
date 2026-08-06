def layerwise_output_mse(
    W: list[list[float]], W_q: list[list[float]], X: list[list[float]]
) -> float:
    m = len(W)
    k = len(W[0])
    n = len(X[0])
    total_sq_err = 0.0
    for i in range(m):
        for j in range(n):
            y_ij = 0.0
            y_q_ij = 0.0
            for p in range(k):
                y_ij += W[i][p] * X[p][j]
                y_q_ij += W_q[i][p] * X[p][j]
            diff = y_ij - y_q_ij
            total_sq_err += diff * diff
    return float(total_sq_err / (m * n))
