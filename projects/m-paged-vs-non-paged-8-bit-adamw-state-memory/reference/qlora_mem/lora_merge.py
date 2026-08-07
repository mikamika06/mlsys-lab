import numpy as np

CODEBOOK_4BIT = np.linspace(-1.0, 1.0, 16, dtype=np.float32)


def dequantize_4bit(qweights: np.ndarray, scales: np.ndarray, block_size: int = 64) -> np.ndarray:
    m, n = qweights.shape
    flat_q = qweights.reshape(-1)
    flat_scales = scales.reshape(-1)
    num_blocks = len(flat_q) // block_size
    dequant = CODEBOOK_4BIT[flat_q].astype(np.float32)
    dequant = dequant.reshape(num_blocks, block_size)
    scaled = dequant * flat_scales[:, np.newaxis]
    return scaled.reshape(m, n)


def merge_lora_into_base(qweights: np.ndarray, scales: np.ndarray, lora_A: np.ndarray, lora_B: np.ndarray, alpha: float, block_size: int = 64) -> np.ndarray:
    w_base = dequantize_4bit(qweights, scales, block_size)
    r = lora_A.shape[0]
    scaling = alpha / float(r)
    delta = (lora_B @ lora_A) * scaling
    return w_base + delta


def quantize_to_4bit(weights: np.ndarray, block_size: int = 64) -> tuple[np.ndarray, np.ndarray]:
    m, n = weights.shape
    flat_w = weights.reshape(-1).astype(np.float32)
    total_elements = len(flat_w)
    num_blocks = total_elements // block_size
    blocks = flat_w.reshape(num_blocks, block_size)
    max_vals = np.max(np.abs(blocks), axis=1)
    scales = np.where(max_vals == 0, 1.0, max_vals).astype(np.float32)
    norm_blocks = blocks / scales[:, np.newaxis]
    diffs = np.abs(norm_blocks[:, :, np.newaxis] - CODEBOOK_4BIT[np.newaxis, np.newaxis, :])
    q_indices = np.argmin(diffs, axis=2).astype(np.uint8)
    qweights = q_indices.reshape(m, n)
    return qweights, scales
