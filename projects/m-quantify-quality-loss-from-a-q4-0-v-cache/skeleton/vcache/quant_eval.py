import numpy as np


def quantize_q4_0(v_matrix, block_size=32):
    """Quantize floating point array to q4_0 blocks."""
    raise NotImplementedError


def dequantize_q4_0(quantized_data, shape, block_size=32):
    """Dequantize q4_0 blocks back to float32."""
    raise NotImplementedError


def evaluate_v_cache_loss(v_matrix, block_size=32):
    """Evaluate relative reconstruction error for q4_0 V cache."""
    raise NotImplementedError
