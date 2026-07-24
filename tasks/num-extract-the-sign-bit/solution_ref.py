import numpy as np

def extract_sign_bit(arr: np.ndarray) -> np.ndarray:
    """Extract the IEEE-754 sign bit from each float32 element.

    Returns a uint8 array: 0 = non-negative, 1 = negative (including -0.0).
    """
    return (arr.view(np.uint32) >> 31).astype(np.uint8)
