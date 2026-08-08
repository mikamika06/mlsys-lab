import numpy as np
from fp8.e4m3 import E4M3_MAX, decode_e4m3, encode_e4m3


def compute_scale(x: np.ndarray) -> float:
    raise NotImplementedError


def quantize_and_descale(
    x: np.ndarray, scale: float
) -> tuple[np.ndarray, np.ndarray]:
    raise NotImplementedError
