import numpy as np


def dynamic_vs_linear_int8_mse(x: np.ndarray):
    """Compare an 8-bit dynamic (non-uniform) map to linear int8 quantization.

    Args:
        x: 1-D float64 NumPy array, not all zeros.

    Returns:
        (mse_dynamic, mse_linear): reconstruction MSE of each scheme.
    """
    raise NotImplementedError('your code here')
