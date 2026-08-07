def migrate_scale(X: list[list[float]], W: list[list[float]], s: list[float]) -> tuple[list[list[float]], list[list[float]]]:
    X_new = []
    for row in X:
        new_row = []
        for j, val in enumerate(row):
            new_row.append(float(val) * float(s[j]))
        X_new.append(new_row)

    W_new = []
    for j, row in enumerate(W):
        s_val = float(s[j])
        new_row = []
        for val in row:
            new_row.append(float(val) / s_val)
        W_new.append(new_row)

    return X_new, W_new
