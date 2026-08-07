from __future__ import annotations

def acceptance_rate(target: list[list[float]], draft: list[list[float]]) -> list[float]:
    n_rows = len(target)
    result = []

    for i in range(n_rows):
        row_sum = 0.0
        n_cols = len(target[i])
        for j in range(n_cols):
            t_val = target[i][j]
            d_val = draft[i][j]
            if t_val < d_val:
                row_sum += t_val
            else:
                row_sum += d_val
        result.append(row_sum)

    return result
