import numpy as np


def compare_fp8_formats(x: np.ndarray) -> tuple[float, float, str]:
    """Cast `x` to the E4M3 and E5M2 FP8 value grids, report each format's
    reconstruction MSE, and which one wins (smaller MSE; ties -> "e4m3").

    Returns:
        (mse_e4m3, mse_e5m2, winner) where winner is "e4m3" or "e5m2".
    """
    raise NotImplementedError('your code here')
