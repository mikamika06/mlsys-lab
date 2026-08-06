import math

def log_softmax(x: list[float] | list[list[float]]) -> list[float] | list[list[float]]:
    """Compute log-softmax along the last axis via the stable x − LSE identity."""
    if not x:
        return []

    is_2d = isinstance(x[0], list)
    rows = x if is_2d else [x]
    out_rows = []

    for row in rows:
        n_cols = len(row)
        if n_cols == 0:
            out_rows.append([])
            continue

        max_val = row[0]
        for j in range(1, n_cols):
            if row[j] > max_val:
                max_val = row[j]

        sum_exp = 0.0
        for j in range(n_cols):
            sum_exp += math.exp(row[j] - max_val)

        lse = max_val + math.log(sum_exp)

        out_row = [row[j] - lse for j in range(n_cols)]
        out_rows.append(out_row)

    if not is_2d:
        return out_rows[0]
    return out_rows
