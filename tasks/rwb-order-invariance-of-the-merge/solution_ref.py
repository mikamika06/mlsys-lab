import numpy as np


def merge_partials(partials):
    arrays = [np.asarray(x, dtype=np.float64) for x in partials]
    if not arrays:
        empty = np.array([], dtype=np.float64)
        return empty, empty

    indexed = sorted(enumerate(arrays), key=lambda item: item[0])
    ordered = [x for _, x in indexed]

    stacked = np.stack(ordered, axis=0).astype(np.float64, copy=False)
    merged = np.sum(stacked, axis=0, dtype=np.float64)

    return merged.copy(), merged.copy()
