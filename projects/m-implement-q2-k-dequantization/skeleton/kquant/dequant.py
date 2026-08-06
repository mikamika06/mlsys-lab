import numpy as np


def dequantize_q2_k(block_bytes: bytes) -> np.ndarray:
    """Dequantizes a single Q2_K block (256 weights) into float32 array."""
    raise NotImplementedError


def reconstruct_q3_k_scales(hmask: bytes, scales_raw: bytes) -> np.ndarray:
    """Reconstructs 16 6-bit scales for a Q3_K block using raw scales and hmask."""
    raise NotImplementedError
