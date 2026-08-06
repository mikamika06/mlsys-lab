import numpy as np


def select_snapkv_indices(attn: np.ndarray, k: int) -> np.ndarray:
    arr = np.asarray(attn, dtype=np.float64)
    rows = arr.shape[0]
    cols = arr.shape[1]

    scores = np.zeros(cols, dtype=np.float64)
    for c in range(cols):
        acc = 0.0
        for r in range(rows):
            acc += arr[r, c]
        scores[c] = acc / float(rows)

    indexed_scores = []
    for c in range(cols):
        indexed_scores.append((scores[c], c))

    def stable_sort_key(item):
        return (-item[0], item[1])

    n = len(indexed_scores)
    for i in range(1, n):
        key = indexed_scores[i]
        j = i - 1
        while j >= 0 and stable_sort_key(indexed_scores[j]) > stable_sort_key(key):
            indexed_scores[j + 1] = indexed_scores[j]
            j -= 1
        indexed_scores[j + 1] = key

    order = np.zeros(k, dtype=np.int64)
    for i in range(k):
        order[i] = indexed_scores[i][1]

    k_len = k
    for i in range(1, k_len):
        key = order[i]
        j = i - 1
        while j >= 0 and order[j] > key:
            order[j + 1] = order[j]
            j -= 1
        order[j + 1] = key

    return order
