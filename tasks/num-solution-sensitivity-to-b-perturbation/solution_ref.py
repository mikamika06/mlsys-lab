import numpy as np


def solution_sensitivity(A, b, delta):
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    delta = np.asarray(delta, dtype=np.float64)

    x = np.linalg.solve(A, b)
    xp = np.linalg.solve(A, b + delta)
    return float(np.linalg.norm(xp - x) / (np.linalg.norm(x) + 1e-12))
