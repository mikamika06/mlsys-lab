import numpy as np


def encode_e4m3(x: np.ndarray) -> np.ndarray:
    """Encode floats to E4M3FN byte patterns (uint8): saturate to +-448,
    round-to-nearest-even to the real 128-point nonnegative grid."""
    raise NotImplementedError('your code here')


def decode_e4m3(codes: np.ndarray) -> np.ndarray:
    """Decode raw E4M3FN byte patterns (uint8) to float values."""
    raise NotImplementedError('your code here')
