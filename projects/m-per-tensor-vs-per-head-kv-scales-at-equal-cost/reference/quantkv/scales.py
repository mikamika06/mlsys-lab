import numpy as np


def compute_scales(tensor: np.ndarray, mode: str, block_size: int = 32) -> np.ndarray:
    """Compute quantization scales for tensor given mode."""
    if mode == "per-tensor":
        mx = np.max(np.abs(tensor))
        scale = mx / 127.0 if mx > 0 else 1.0
        return np.array([scale], dtype=np.float32)
    elif mode == "per-head":
        reshaped = tensor.reshape(tensor.shape[0], tensor.shape[1], -1)
        mxs = np.max(np.abs(reshaped), axis=(0, 2))
        scales = mxs / 127.0
        scales[scales == 0] = 1.0
        return scales.astype(np.float32)
    elif mode == "per-block":
        flat = tensor.flatten()
        padded_len = ((len(flat) + block_size - 1) // block_size) * block_size
        padded = np.zeros(padded_len, dtype=tensor.dtype)
        padded[:len(flat)] = flat
        blocks = padded.reshape(-1, block_size)
        mxs = np.max(np.abs(blocks), axis=1)
        scales = mxs / 127.0
        scales[scales == 0] = 1.0
        return scales.astype(np.float32)
    else:
        raise ValueError(f"unknown mode {mode}")
