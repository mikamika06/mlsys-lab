import numpy as np


def build_tree_attention_mask(parents):
    n = len(parents)
    mask = np.zeros((n, n), dtype=np.int64)

    for i in range(n):
        node = i
        while node != -1:
            mask[i, node] = 1
            node = parents[node]

    return mask
