import numpy as np


def log_likelihood(probs: np.ndarray) -> float:
    """Log-likelihood computed by multiplying probabilities then taking the log."""
    probs = np.asarray(probs, dtype=np.float64)
    return float(np.log(np.prod(probs)))
