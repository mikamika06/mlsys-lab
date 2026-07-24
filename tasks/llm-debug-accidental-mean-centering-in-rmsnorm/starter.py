import numpy as np

def rms_norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """
    Incorrect implementation that subtracts the mean before normalizing.
    This mimics a LayerNorm style centering and is not what RMSNorm should do.
    """
    # TODO: remove the mean subtraction to implement true RMSNorm
    x_centered = x - np.mean(x, axis=-1, keepdims=True)
    rms = np.sqrt(np.mean(x_centered**2, axis=-1, keepdims=True) + eps)
    return x_centered / rms
