import numpy as np


def quantize_int8_per_channel(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Quantizes 2D weight matrix to INT8 per-channel (out_channels, in_channels)."""
    raise NotImplementedError


def dequantize_int8_per_channel(q: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Dequantizes INT8 per-channel weight matrix to FP32."""
    raise NotImplementedError


def quantize_int4_blockwise(w: np.ndarray, block_size: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Quantizes 2D weight matrix to INT4 blockwise (packed into uint8 or int8) with FP16/FP32 scale per block."""
    raise NotImplementedError


def dequantize_int4_blockwise(q: np.ndarray, scales: np.ndarray, block_size: int = 32) -> np.ndarray:
    """Dequantizes INT4 blockwise weight matrix to FP32."""
    raise NotImplementedError
