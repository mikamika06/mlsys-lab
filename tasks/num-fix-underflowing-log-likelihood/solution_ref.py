import numpy as np


def log_likelihood(probs: np.ndarray) -> float:
    """Log-likelihood via sum of logs (avoids underflow from multiplying probabilities directly)."""
    probs = np.asarray(probs, dtype=np.float64)
    return float(np.sum(np.log(probs)))
