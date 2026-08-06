import numpy as np


def compare_retention(attention, window_size, budget, needle_index):
    attention = np.asarray(attention, dtype=np.float64)
    rows = attention.shape[0]
    n = attention.shape[1]

    mass = [0.0] * n
    for j in range(n):
        s = 0.0
        for i in range(rows):
            s += float(attention[i, j])
        mass[j] = s

    streaming_set = set(range(min(2, n)))
    streaming_set.update(range(max(0, n - window_size), n))
    streaming = sorted(streaming_set)

    ranked = sorted(range(n), key=lambda i: (-mass[i], i))
    h2o = sorted(ranked[: min(budget, n)])

    streaming_mass = 0.0
    for idx in streaming:
        streaming_mass += mass[idx]

    h2o_mass = 0.0
    for idx in h2o:
        h2o_mass += mass[idx]

    return {
        "streaming_retained": streaming,
        "h2o_retained": h2o,
        "streaming_keeps_needle": needle_index in streaming,
        "h2o_keeps_needle": needle_index in h2o,
        "streaming_mass": float(streaming_mass),
        "h2o_mass": float(h2o_mass),
    }
