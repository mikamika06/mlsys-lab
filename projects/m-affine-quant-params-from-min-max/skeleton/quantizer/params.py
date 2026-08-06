import numpy as np


def calc_affine_params(val_min: float, val_max: float, qmin: int = 0, qmax: int = 255) -> tuple[float, int]:
    raise NotImplementedError


def quantize(x: np.ndarray, scale: float, zero_point: int, qmin: int = 0, qmax: int = 255) -> np.ndarray:
    raise NotImplementedError


def dequantize(q: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    raise NotImplementedError
