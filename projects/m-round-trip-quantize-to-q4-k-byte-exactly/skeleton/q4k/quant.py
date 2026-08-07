import numpy as np


def quantize_q4_k_superblock(weights: np.ndarray) -> bytes:
    raise NotImplementedError


def dequantize_q4_k_superblock(data: bytes) -> np.ndarray:
    raise NotImplementedError


def round_trip_q4_k(weights: np.ndarray) -> tuple[bytes, np.ndarray]:
    raise NotImplementedError
