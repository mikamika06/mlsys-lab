def accumulate_grad(micro_batches: list[tuple[list[list[float]], list[float]]], w: list[float]) -> list[float]:
    """
    Gradient of mean squared error loss w.r.t. w, computed by accumulating
    each micro-batch's contribution and normalizing by the TOTAL example
    count -- exactly the gradient a single large batch would produce.
    """
    D = len(w)
    total = [0.0] * D
    N = 0

    for X_i, y_i in micro_batches:
        b_i = len(X_i)
        N += b_i

        r_i = [0.0] * b_i
        for i in range(b_i):
            row_dot = 0.0
            for j in range(D):
                row_dot += X_i[i][j] * w[j]
            r_i[i] = row_dot - y_i[i]

        for j in range(D):
            col_sum = 0.0
            for i in range(b_i):
                col_sum += X_i[i][j] * r_i[i]
            total[j] += col_sum

    return [(2.0 / N) * val for val in total]
