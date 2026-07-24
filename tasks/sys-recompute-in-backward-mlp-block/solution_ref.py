import numpy as np

def checkpoint_forward(x: np.ndarray,
                       W1: np.ndarray, b1: np.ndarray,
                       W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    z = W1 @ x + b1
    a = np.maximum(0, z)
    return W2 @ a + b2

def checkpoint_backward(dy: np.ndarray,
                        x: np.ndarray,
                        W1: np.ndarray, b1: np.ndarray,
                        W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    z = W1 @ x + b1
    mask = (z > 0).astype(float)
    da = W2.T @ dy
    dz = da * mask
    return W1.T @ dz
