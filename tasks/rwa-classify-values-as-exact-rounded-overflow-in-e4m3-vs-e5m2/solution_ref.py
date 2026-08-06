import numpy as np


def classify_fp8(values: np.ndarray, grid: np.ndarray, max_finite: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)
    grid_set = set(grid.flat)

    labels = []
    for val in values.flat:
        av = abs(val)
        if av > max_finite:
            labels.append("overflow")
        elif av in grid_set:
            labels.append("exact")
        else:
            labels.append("rounded")

    return np.array(labels, dtype="<U8").reshape(values.shape)
