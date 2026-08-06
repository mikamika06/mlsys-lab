import numpy as np


def find_optimal_k(alpha, c, max_k):
    """Find draft length k maximizing expected speedup given acceptance rate and cost ratio."""
    best_k = 1
    best_speedup = -1.0
    speedups = {}

    for k in range(1, max_k + 1):
        if abs(alpha - 1.0) < 1e-9:
            tokens = float(k + 1)
        else:
            tokens = (1.0 - alpha ** (k + 1)) / (1.0 - alpha)

        speedup = tokens / (1.0 + k * c)
        speedups[k] = float(speedup)

        if speedup > best_speedup + 1e-12:
            best_speedup = speedup
            best_k = k

    return best_k, speedups
