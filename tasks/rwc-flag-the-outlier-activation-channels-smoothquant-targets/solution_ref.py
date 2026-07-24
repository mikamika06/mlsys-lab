import numpy as np

def flag_outliers(X: np.ndarray, factor: float = 3.0) -> np.ndarray:
    """
    Return a boolean mask indicating channels whose maximum absolute activation
    exceeds `factor` times the median of all channel maxima.
    """
    # Compute per‑channel max |X|
    m = np.max(np.abs(X), axis=0)
    # Median of these maxima
    med = np.median(m)
    # Boolean mask of outliers
    return m > (factor * med)
