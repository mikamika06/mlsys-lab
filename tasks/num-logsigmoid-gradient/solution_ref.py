import numpy as np


def logsigmoid_with_grad(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    value = -np.logaddexp(0.0, -x)
    grad = 1.0 / (1.0 + np.exp(x))
    return value, grad
