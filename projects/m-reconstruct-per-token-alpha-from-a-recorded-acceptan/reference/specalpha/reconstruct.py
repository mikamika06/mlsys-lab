import numpy as np


def reconstruct_alphas(histogram, max_k):
    total = float(np.sum(list(histogram.values())))
    if total <= 0:
        return [0.0] * max_k
    probs = []
    surviving = total
    for k in range(1, max_k + 1):
        count = histogram.get(k, 0.0)
        if surviving <= 0:
            probs.append(0.0)
        else:
            p = min(1.0, max(0.0, count / surviving))
            probs.append(p)
            surviving -= count
    return probs
