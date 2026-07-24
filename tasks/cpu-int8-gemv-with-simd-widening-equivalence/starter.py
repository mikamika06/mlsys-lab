import numpy as np


def int8_gemv(A: np.ndarray, x: np.ndarray):
    """Compute y = A @ x with int8 inputs widened to int32. Return (y, access_trace)."""
    raise NotImplementedError("implement int8 GEMV + access trace")
