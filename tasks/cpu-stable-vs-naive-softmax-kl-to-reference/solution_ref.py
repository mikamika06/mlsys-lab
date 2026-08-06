import math


def stable_softmax_kernel(logits: list[list[float]]) -> tuple[list[list[float]], list[int]]:
    n = len(logits)
    d = len(logits[0])
    result = [[0.0 for _ in range(d)] for _ in range(n)]

    for r in range(n):
        max_val = logits[r][0]
        for c in range(1, d):
            val = logits[r][c]
            if val > max_val:
                max_val = val

        row_sum = 0.0
        for c in range(d):
            exp_val = math.exp(logits[r][c] - max_val)
            result[r][c] = exp_val
            row_sum += exp_val

        for c in range(d):
            result[r][c] = result[r][c] / row_sum

    trace = []
    for r in range(n):
        for c in range(d):
            trace.append((r * d + c) * 8)
    for r in range(n):
        for c in range(d):
            trace.append((r * d + c) * 8)

    return result, trace
