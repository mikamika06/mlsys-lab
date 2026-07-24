import numpy as np


def _square(v: float) -> float:
    return v * v


def sum_squares_call_heavy(x: np.ndarray) -> float:
    """sum(x_i**2), calling a helper function once per element."""
    total = 0.0
    for xi in x:
        total += _square(float(xi))
    return total


def sum_squares_inlined(x: np.ndarray) -> float:
    """sum(x_i**2), with the squaring inlined directly in the loop body."""
    total = 0.0
    for xi in x:
        v = float(xi)
        total += v * v
    return total
