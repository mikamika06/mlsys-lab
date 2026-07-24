import numpy as np


def classify_fp8(values: np.ndarray, grid: np.ndarray, max_finite: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)
    av = np.abs(values)

    labels = np.empty(values.shape, dtype="<U8")
    overflow = av > max_finite
    labels[overflow] = "overflow"

    inrange = ~overflow
    exact_mask = np.isin(av, grid)
    labels[inrange & exact_mask] = "exact"
    labels[inrange & ~exact_mask] = "rounded"

    return labels
