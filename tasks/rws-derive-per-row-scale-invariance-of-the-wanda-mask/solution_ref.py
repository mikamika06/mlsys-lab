def wanda_mask(W: list[list[float]], col_norms: list[float], keep_ratio: float) -> list[list[bool]]:
    rows = len(W)
    cols = len(W[0]) if rows > 0 else 0
    k = max(1, int(round(cols * keep_ratio)))

    mask = [[False for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        row_data = []
        for j in range(cols):
            val = W[i][j]
            abs_val = val if val >= 0.0 else -val
            score = abs_val * col_norms[j]
            row_data.append((j, score))

        sorted_row = sorted(row_data, key=lambda x: -x[1])
        for idx in range(k):
            j_idx = sorted_row[idx][0]
            mask[i][j_idx] = True

    return mask
