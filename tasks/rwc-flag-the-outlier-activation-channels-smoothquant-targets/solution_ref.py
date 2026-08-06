import numpy as np

def flag_outliers(X: np.ndarray, factor: float = 3.0) -> np.ndarray:
    """
    Return a boolean mask indicating channels whose maximum absolute activation
    exceeds `factor` times the median of all channel maxima.
    """
    rows = X.shape[0]
    cols = X.shape[1]
    
    m = np.zeros(cols, dtype=X.dtype)
    for j in range(cols):
        max_val = 0.0
        if rows > 0:
            first_val = X[0, j]
            if first_val < 0.0:
                max_val = -first_val
            else:
                max_val = first_val
        for i in range(1, rows):
            val = X[i, j]
            if val < 0.0:
                val = -val
            if val > max_val:
                max_val = val
        m[j] = max_val

    sorted_m = np.zeros(cols, dtype=m.dtype)
    for j in range(cols):
        sorted_m[j] = m[j]
    
    for i in range(cols):
        for j in range(cols - 1 - i):
            if sorted_m[j] > sorted_m[j + 1]:
                temp = sorted_m[j]
                sorted_m[j] = sorted_m[j + 1]
                sorted_m[j + 1] = temp

    if cols % 2 == 1:
        med = sorted_m[cols // 2]
    else:
        med = (sorted_m[cols // 2 - 1] + sorted_m[cols // 2]) / 2.0

    threshold = factor * med
    result = np.zeros(cols, dtype=bool)
    for j in range(cols):
        result[j] = m[j] > threshold

    return result
