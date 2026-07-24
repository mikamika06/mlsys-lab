import numpy as np


def compare_retention(attention, window_size, budget, needle_index):
    attention = np.asarray(attention, dtype=np.float64)
    n = attention.shape[1]
    mass = np.sum(attention, axis=0)

    streaming = set(range(min(2, n)))
    streaming.update(range(max(0, n - window_size), n))
    streaming = sorted(streaming)

    ranked = sorted(range(n), key=lambda i: (-mass[i], i))
    h2o = sorted(ranked[:min(budget, n)])

    return {
        "streaming_retained": streaming,
        "h2o_retained": h2o,
        "streaming_keeps_needle": needle_index in streaming,
        "h2o_keeps_needle": needle_index in h2o,
        "streaming_mass": float(np.sum(mass[streaming])),
        "h2o_mass": float(np.sum(mass[h2o])),
    }
