import numpy as np


def best_h_central_diff(f, fprime, x: float, h_grid: np.ndarray) -> float:
    h_grid = np.asarray(h_grid, dtype=np.float64)
    true = fprime(x)

    best_h = None
    best_err = np.inf
    for h in h_grid:
        approx = (f(x + h) - f(x - h)) / (2.0 * h)
        err = abs(approx - true) / (abs(true) + 1e-300)
        if err < best_err:
            best_err = err
            best_h = float(h)
    return best_h
