import numpy as np


def q6_k_dequantize(d: float, scales: np.ndarray, ql: np.ndarray, qh: np.ndarray) -> np.ndarray:
    """Reconstruct the 256 float32 values of a ggml Q6_K super-block.

    Args:
        d: super-block scale (already a plain float, not raw fp16 bits).
        scales: 16 signed int8 sub-scales.
        ql: 128 uint8 bytes, low nibbles of the 6-bit codes.
        qh: 64 uint8 bytes, high 2-bit pairs of the 6-bit codes.

    Returns:
        np.ndarray of shape (256,), dtype float32.
    """
    raise NotImplementedError('your code here')
