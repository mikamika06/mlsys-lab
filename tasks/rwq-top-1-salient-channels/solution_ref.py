import math


def top_salient_channels(X: list[list[float]], frac: float = 0.01) -> list[int]:
    """AWQ-style salient-channel selection: the top `frac` fraction of
    channels (columns) by mean absolute calibration activation.

    Returns a list of integer channel indices, length ceil(frac * C)
    (minimum 1), the channels with the largest mean(|X|).
    """
    N = len(X)
    C = len(X[0]) if N > 0 else 0

    scores = []
    for c in range(C):
        total = 0.0
        for r in range(N):
            val = X[r][c]
            if val < 0.0:
                total -= val
            else:
                total += val
        scores.append(total / N)

    k = max(1, int(math.ceil(frac * C)))

    indexed_scores = []
    for c in range(C):
        indexed_scores.append((scores[c], c))

    def sort_key(item):
        return (-item[0], item[1])

    n = len(indexed_scores)
    for i in range(1, n):
        key = indexed_scores[i]
        j = i - 1
        while j >= 0 and sort_key(indexed_scores[j]) > sort_key(key):
            indexed_scores[j + 1] = indexed_scores[j]
            j -= 1
        indexed_scores[j + 1] = key

    order = []
    for i in range(k):
        order.append(indexed_scores[i][1])

    return order
