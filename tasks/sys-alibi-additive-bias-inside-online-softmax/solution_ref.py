import math

def alibi_online_softmax(scores: list[list[float]], slopes: list[float]) -> list[list[float]]:
    """Online softmax with ALiBi bias integrated into the streaming loop."""
    n = len(scores)
    probs = [[0.0] * n for _ in range(n)]

    for i in range(n):
        m = slopes[i]
        running_max = -float('inf')
        running_sum = 0.0

        for j in range(n):
            v_j = float(scores[i][j]) + float(m) * (i - j)
            if v_j > running_max:
                running_sum = running_sum * math.exp(running_max - v_j) + 1.0
                running_max = v_j
            else:
                running_sum += math.exp(v_j - running_max)

        for j in range(n):
            v_j = float(scores[i][j]) + float(m) * (i - j)
            probs[i][j] = math.exp(v_j - running_max) / running_sum

    return probs
