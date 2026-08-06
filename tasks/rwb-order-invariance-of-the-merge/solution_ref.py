import numpy as np


def merge_partials(partials):
    arrays = [np.asarray(x, dtype=np.float64) for x in partials]
    if not arrays:
        empty = np.array([], dtype=np.float64)
        return empty, empty

    indexed = sorted(enumerate(arrays), key=lambda item: item[0])
    ordered = [x for _, x in indexed]

    num_rows = len(ordered)
    num_cols = len(ordered[0])

    merged_list = []
    for j in range(num_cols):
        col_sum = 0.0
        for i in range(num_rows):
            col_sum += float(ordered[i][j])
        merged_list.append(col_sum)

    merged = np.array(merged_list, dtype=np.float64)

    return merged.copy(), merged.copy()
