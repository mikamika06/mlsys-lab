import numpy as np


def per_block_int8_quant_matmul(
    A: np.ndarray, B: np.ndarray, block_size: int = 32
) -> np.ndarray:
    """Perform per-block INT8 quantization, matrix multiplication, and dequantization."""
    raise NotImplementedError
