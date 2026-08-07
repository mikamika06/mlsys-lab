def pruned_shell_forward(W, b, x, keep_rows, keep_cols):
    Wp = [[W[i][j] for j in keep_cols] for i in keep_rows]
    bp = [b[i] for i in keep_rows]
    xp = [x[j] for j in keep_cols]

    result = []
    for row in Wp:
        dot_prod = sum(w_val * x_val for w_val, x_val in zip(row, xp))
        result.append(dot_prod)

    return [r + b_val for r, b_val in zip(result, bp)]
