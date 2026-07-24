import numpy as np


def best_h_central_diff(f, fprime, x: float, h_grid: np.ndarray) -> float:
    """Search h_grid for the step size that minimizes the central-difference
    relative error against fprime(x), and return that h (an actual element
    of h_grid).
    """
    raise NotImplementedError('your code here')
