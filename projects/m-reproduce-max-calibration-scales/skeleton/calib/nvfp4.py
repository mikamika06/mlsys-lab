import numpy as np


def quantize_nvfp4_block(
    tensor: np.ndarray, block_size: int = 16
) -> tuple[np.ndarray, np.ndarray]:
    raise NotImplementedError


def dequantize_nvfp4_block(
    codes: np.ndarray, scales: np.ndarray, block_size: int = 16
) -> np.ndarray:
    raise NotImplementedError


def nvfp4_round_trip(tensor: np.ndarray, block_size: int = 16) -> np.ndarray:
    raise NotImplementedError
