import numpy as np


def optimal_gamma(alpha: float, c: float, max_gamma: int = 8) -> int:
    gammas = np.arange(1, max_gamma + 1)
    expected_accepted = (1.0 - alpha ** (gammas + 1)) / (1.0 - alpha)
    cost = (1.0 + c * gammas) / expected_accepted
    return int(np.argmin(cost) + 1)
