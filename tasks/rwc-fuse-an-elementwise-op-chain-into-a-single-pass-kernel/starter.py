import numpy as np


def fused_elementwise_chain(X: np.ndarray, bias: np.ndarray, residual: np.ndarray, scale: float) -> np.ndarray:
    """
    Compute, as a single fused elementwise expression:
        h = X + bias
        h = gelu_tanh(h)     # 0.5*h*(1 + tanh(sqrt(2/pi)*(h + 0.044715*h**3)))
        h = h + residual
        h = h * scale
    Returns an array the same shape as X. See task.md.
    """
    raise NotImplementedError('your code here')
