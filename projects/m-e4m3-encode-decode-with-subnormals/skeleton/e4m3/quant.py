import numpy as np


def encode_e4m3(arr: np.ndarray) -> np.ndarray:
    """Encode float32 array into uint8 representing FP8 E4M3."""
    raise NotImplementedError


def decode_e4m3(bytes_arr: np.ndarray) -> np.ndarray:
    """Decode uint8 array of FP8 E4M3 values back to float32."""
    raise NotImplementedError
