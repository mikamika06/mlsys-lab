import numpy as np


def allocate_ranks(errors, budget):
    errors = np.asarray(errors, dtype=np.float32)
    n = len(errors)
    if n == 0:
        return []
    ranks = np.zeros(n, dtype=int)
    for _ in range(int(budget)):
        marginal_gains = errors / (ranks + 1.0)
        best_idx = int(np.argmax(marginal_gains))
        ranks[best_idx] += 1
    return ranks.tolist()
