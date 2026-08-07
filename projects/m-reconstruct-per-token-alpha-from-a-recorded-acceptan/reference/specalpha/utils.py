import numpy as np


def normalize_histogram(histogram):
    total = float(np.sum(list(histogram.values())))
    if total <= 0:
        return {k: 0.0 for k in histogram}
    return {k: v / total for k, v in histogram.items()}
