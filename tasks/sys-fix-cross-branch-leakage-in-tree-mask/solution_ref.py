import numpy as np


def build_tree_mask(parents):
    n = len(parents)
    mask = np.zeros((n, n), dtype=np.int8)

    for i in range(n):
        node = i
        while node != -1:
            mask[i, node] = 1
            node = parents[node]

    return mask
