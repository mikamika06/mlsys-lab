from typing import Tuple
import numpy as np


def quantize_per_tensor_int8(X: np.ndarray) -> Tuple[np.ndarray, float]:
    max_val = np.max(np.abs(X))
    scale = float(max_val / 127.0) if max_val > 0 else 1.0
    q = np.clip(np.round(X / scale), -128, 127).astype(np.int8)
    return q, scale


def quantize_per_block_int8(X: np.ndarray, block_size: int) -> Tuple[np.ndarray, np.ndarray]:
    M, N = X.shape
    padded_M = ((M + block_size - 1) // block_size) * block_size
    padded_N = ((N + block_size - 1) // block_size) * block_size
    X_padded = np.zeros((padded_M, padded_N), dtype=X.dtype)
    X_padded[:M, :N] = X

    blocks = X_padded.reshape(padded_M // block_size, block_size, padded_N // block_size, block_size)
    blocks = blocks.transpose(0, 2, 1, 3)

    max_vals = np.max(np.abs(blocks), axis=(2, 3), keepdims=True)
    scales = max_vals / 127.0
    scales = np.where(scales == 0, 1.0, scales)

    q_blocks = np.clip(np.round(blocks / scales), -128, 127).astype(np.int8)

    q_out = q_blocks.transpose(0, 2, 1, 3).reshape(padded_M, padded_N)[:M, :N]
    scales_out = scales.squeeze((-1, -2))
    return q_out, scales_out
