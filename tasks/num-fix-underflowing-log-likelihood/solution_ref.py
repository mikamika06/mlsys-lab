import math


def log_likelihood(probs: list[float]) -> float:
    """Log-likelihood via sum of logs (avoids underflow from multiplying probabilities directly)."""
    total = 0.0
    for p in probs:
        total += math.log(p)
    return total
