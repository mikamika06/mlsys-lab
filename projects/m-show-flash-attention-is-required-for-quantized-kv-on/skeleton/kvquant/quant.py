import numpy as np


def quantize_q8_0(x: np.ndarray, block_size: int = 32) -> dict:
    raise NotImplementedError


def dequantize_q8_0(qdict: dict) -> np.ndarray:
    raise NotImplementedError


def max_abs_error_bound(x: np.ndarray, block_size: int = 32) -> float:
    raise NotImplementedError
