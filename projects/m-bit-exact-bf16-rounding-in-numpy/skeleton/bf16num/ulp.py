import numpy as np


def get_dtype_max(dtype_str: str) -> float:
    """Returns the maximum representable finite positive value for a given low-precision dtype."""
    raise NotImplementedError


def compute_ulp(x: np.ndarray, dtype_str: str) -> np.ndarray:
    """Computes Unit in the Last Place (ULP) for each element of x for the given dtype."""
    raise NotImplementedError
