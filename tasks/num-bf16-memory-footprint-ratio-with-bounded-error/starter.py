import numpy as np


def pack_bf16(x: np.ndarray) -> np.ndarray:
    """float32 array -> uint16 array of bf16 codes, same shape, round-to-nearest-even."""
    raise NotImplementedError('your code here')


def unpack_bf16(codes: np.ndarray) -> np.ndarray:
    """uint16 array of bf16 codes -> the float32 values they denote, same shape."""
    raise NotImplementedError('your code here')
