import numpy as np

def logsumexp(x: np.ndarray) -> float:
    """Stable computation of the log‑sum‑exp of a 1‑D array."""
    m = np.max(x)
    return float(m + np.log(np.sum(np.exp(x - m))))
