import numpy as np

def nesterov_minimize(Q: np.ndarray,
                      c: np.ndarray,
                      x0: np.ndarray,
                      lr: float,
                      beta: float,
                      T: int) -> np.ndarray:
    """Correct implementation of Nesterov accelerated gradient for a quadratic."""
    x_prev = x0.copy()
    x_curr = x0.copy()
    for _ in range(T):
        y = x_curr + beta * (x_curr - x_prev)
        grad = Q @ y - c
        x_next = y - lr * grad
        x_prev, x_curr = x_curr, x_next
    return x_curr
