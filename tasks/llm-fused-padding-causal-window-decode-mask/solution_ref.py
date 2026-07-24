import numpy as np

def fused_decode_mask(padding_mask: np.ndarray, window_size: int) -> np.ndarray:
    """
    Return a boolean mask of shape (B, T, T) that enforces padding,
    causal, and sliding‑window constraints for decoding.
    """
    B, T = padding_mask.shape
    rows = np.arange(T).reshape(-1, 1)
    cols = np.arange(T).reshape(1, -1)
    # Windowed causal mask: j <= i and i-j < window_size
    window = (cols <= rows) & (rows - cols < window_size)
    # Broadcast across batch
    mask = window[None, :, :] & padding_mask[:, None, :]
    # Zero out rows where target is padded
    mask[~padding_mask[:, :, None]] = False
    return mask
