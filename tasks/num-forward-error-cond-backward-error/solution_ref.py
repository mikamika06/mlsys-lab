import numpy as np


def forward_error_bound(A: np.ndarray, b: np.ndarray, delta_b: np.ndarray) -> tuple[float, float]:
    x = np.linalg.solve(A, b)
    x_hat = np.linalg.solve(A, b + delta_b)

    forward_error = np.linalg.norm(x_hat - x) / (np.linalg.norm(x) + 1e-12)
    backward_error = np.linalg.norm(delta_b) / (np.linalg.norm(b) + 1e-12)
    bound = np.linalg.cond(A) * backward_error

    return float(forward_error), float(bound)
