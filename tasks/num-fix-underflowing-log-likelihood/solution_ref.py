import math
import numpy as np


def log_likelihood(probs: np.ndarray) -> float:
    """Log-likelihood via sum of logs (avoids underflow from multiplying probabilities directly)."""
    probs = np.asarray(probs, dtype=np.float64)
    total = 0.0
    for p in probs:
        total += math.log(p)
    return float(total)
