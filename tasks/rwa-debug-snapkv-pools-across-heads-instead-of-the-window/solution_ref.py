def select_snapkv_indices(attn: list[list[float]], k: int) -> list[int]:
    h = len(attn)
    w = len(attn[0])
    scores = [0.0] * w
    for j in range(w):
        col_sum = 0.0
        for i in range(h):
            col_sum += attn[i][j]
        scores[j] = col_sum / h

    indexed_scores = [(scores[j], j) for j in range(w)]
    indexed_scores.sort(key=lambda x: (-x[0], x[1]))

    top_indices = [x[1] for x in indexed_scores[:k]]
    top_indices.sort()
    return top_indices
