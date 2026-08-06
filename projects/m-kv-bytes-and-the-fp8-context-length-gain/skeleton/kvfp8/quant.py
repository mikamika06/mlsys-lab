import numpy as np


def quantize_fp8_per_head(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raise NotImplementedError


def dequantize_fp8_per_head(q: np.ndarray, scale: np.ndarray) -> np.ndarray:
    raise NotImplementedError
