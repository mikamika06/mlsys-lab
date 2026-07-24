import numpy as np


def reduce_scatter_sum(chunks):
    ranks = len(chunks)
    result = []
    for i in range(ranks):
        total = np.zeros_like(chunks[0][i], dtype=np.float64)
        for r in range(ranks):
            total = total + chunks[r][i]
        result.append(total)
    return result
