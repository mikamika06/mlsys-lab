import numpy as np


def classify_fp8(values: np.ndarray, grid: np.ndarray, max_finite: float) -> np.ndarray:
    """Label each value 'exact' / 'rounded' / 'overflow' against a format's grid.

    values: array of values to classify.
    grid: ascending 1-D array of representable nonnegative magnitudes
        (grid[-1] == max_finite).
    max_finite: largest finite representable magnitude.

    Returns a same-shape array of string labels, using |v|:
      - "exact"    if |v| is a member of grid
      - "rounded"  if |v| <= max_finite but not a member of grid
      - "overflow" if |v| > max_finite
    """
    raise NotImplementedError('your code here')
