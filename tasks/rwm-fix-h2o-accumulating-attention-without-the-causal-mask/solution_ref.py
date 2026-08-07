import math


def select_heavy_hitters(attn_scores: list[list[float]], budget: int) -> list[int]:
    n = len(attn_scores)

    masked = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if j <= i:
                masked[i][j] = attn_scores[i][j]
            else:
                masked[i][j] = -float('inf')

    probs = [[0.0] * n for _ in range(n)]
    for i in range(n):
        max_val = max(masked[i])
        shifted = [val - max_val for val in masked[i]]
        exps = [math.exp(val) if not math.isinf(val) else 0.0 for val in shifted]
        sum_exps = sum(exps)
        if sum_exps > 0:
            probs[i] = [e / sum_exps for e in exps]
        else:
            probs[i] = [0.0] * n

    importance = [0.0] * n
    for j in range(n):
        col_sum = 0.0
        for i in range(n):
            col_sum += probs[i][j]
        importance[j] = col_sum

    order = sorted(range(n), key=lambda i: (-importance[i], i))
    return order[:budget]
