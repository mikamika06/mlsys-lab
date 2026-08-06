import numpy as np

NF4_LEVELS = np.array([
    -1.0, -0.6961928, -0.52507305, -0.39491749,
    -0.28444138, -0.18477343, -0.09105004, 0.0,
    0.0795803, 0.1609302, 0.2461123, 0.33791524,
    0.44070983, 0.562617, 0.72295683, 1.0
], dtype=np.float32)


def simulate_nf4_cycles(tensor: np.ndarray, cycles: int = 10, block_size: int = 64) -> np.ndarray:
    """Simulate compounding error from repeated NF4 quantize-dequantize cycles."""
    current = tensor.astype(np.float32)
    orig_shape = current.shape
    flat = current.reshape(-1)
    n = flat.size

    pad_len = (block_size - (n % block_size)) % block_size
    if pad_len > 0:
        padded = np.pad(flat, (0, pad_len), mode="constant")
    else:
        padded = flat.copy()

    blocks = padded.reshape(-1, block_size)

    for _ in range(cycles):
        max_abs = np.max(np.abs(blocks), axis=1, keepdims=True)
        max_abs = np.float16(max_abs).astype(np.float32)
        scale = np.where(max_abs == 0.0, 1.0, max_abs)

        scaled = blocks / scale
        diffs = np.abs(scaled[:, :, None] - NF4_LEVELS[None, None, :])
        indices = np.argmin(diffs, axis=-1)

        dequant = NF4_LEVELS[indices] * scale
        blocks = np.float16(dequant).astype(np.float32)

    result_flat = blocks.reshape(-1)[:n]
    return result_flat.reshape(orig_shape)
