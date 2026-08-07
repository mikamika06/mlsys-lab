import numpy as np


def quantize_blockwise(weights: np.ndarray, block_size: int, fmt: str = "nf4"):
    raise NotImplementedError


def dequantize_blockwise(quantized: np.ndarray, scales: np.ndarray, block_size: int, fmt: str = "nf4"):
    raise NotImplementedError
