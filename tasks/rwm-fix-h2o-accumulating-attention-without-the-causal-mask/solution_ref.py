import math
import numpy as np


def select_heavy_hitters(attn_scores: np.ndarray, budget: int) -> np.ndarray:
    scores = np.asarray(attn_scores, dtype=np.float64)
    n = scores.shape[0]

    masked = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if j > i:
                masked[i][j] = -math.inf
            else:
                masked[i][j] = scores[i, j]

    shifted = [[0.0] * n for _ in range(n)]
    for i in range(n):
        row_max = -math.inf
        for j in range(n):
            if masked[i][j] > row_max:
                row_max = masked[i][j]
        for j in range(n):
            if masked[i][j] == -math.inf:
                shifted[i][j] = -math.inf
            else:
                shifted[i][j] = masked[i][j] - row_max

    exp_scores = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if shifted[i][j] == -math.inf:
                exp_scores[i][j] = 0.0
            else:
                exp_scores[i][j] = math.exp(shifted[i][j])

    probs = [[0.0] * n for _ in range(n)]
    for i in range(n):
        row_sum = 0.0
        for j in range(n):
            row_sum += exp_scores[i][j]
        for j in range(n):
            if row_sum == 0.0:
                probs[i][j] = 0.0
            else:
                probs[i][j] = exp_scores[i][j] / row_sum

    importance = [0.0] * n
    for j in range(n):
        col_sum = 0.0
        for i in range(n):
            col_sum += probs[i][j]
        importance[j] = col_sum

    order = sorted(range(n), key=lambda i: (-importance[i], i))
    return np.asarray(order[:budget], dtype=np.int64)
