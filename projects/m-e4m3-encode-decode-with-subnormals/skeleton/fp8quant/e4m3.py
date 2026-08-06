import numpy as np


def encode_e4m3(x: np.ndarray) -> np.ndarray:
    """Encodes float32 array into FP8 E4M3 uint8 bit patterns with subnormals."""
    raise NotImplementedError


def decode_e4m3(u: np.ndarray) -> np.ndarray:
    """Decodes FP8 E4M3 uint8 bit patterns to float32 values."""
    raise NotImplementedError
