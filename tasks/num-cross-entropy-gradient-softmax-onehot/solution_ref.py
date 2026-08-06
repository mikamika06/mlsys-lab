import math


def cross_entropy_backward(logits: list[list[float]], labels: list[int]) -> list[list[float]]:
    """Gradient of the mean softmax cross-entropy w.r.t. ``logits``."""
    n = len(logits)
    c = len(logits[0])

    out = [[0.0 for _ in range(c)] for _ in range(n)]

    for i in range(n):
        row_max = float(logits[i][0])
        for j in range(1, c):
            val = float(logits[i][j])
            if val > row_max:
                row_max = val

        e_sum = 0.0
        for j in range(c):
            val = math.exp(float(logits[i][j]) - row_max)
            out[i][j] = val
            e_sum += val

        target = int(labels[i])
        for j in range(c):
            p = out[i][j] / e_sum
            if j == target:
                p -= 1.0
            out[i][j] = p / float(n)

    return out
