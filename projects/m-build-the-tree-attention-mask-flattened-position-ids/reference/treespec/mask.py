import numpy as np


def build_tree_mask_and_positions(parents, kv_len):
    n = len(parents)
    position_ids = np.zeros(n, dtype=np.int32)
    for i in range(n):
        p = parents[i]
        if p == -1:
            position_ids[i] = kv_len
        else:
            position_ids[i] = position_ids[p] + 1

    mask = np.zeros((n, kv_len + n), dtype=bool)
    for i in range(n):
        mask[i, :kv_len] = True
        curr = i
        path = []
        while curr != -1:
            path.append(curr)
            curr = parents[curr]
        path.reverse()
        for idx, node in enumerate(path):
            mask[i, kv_len + node] = True
    return mask, position_ids
