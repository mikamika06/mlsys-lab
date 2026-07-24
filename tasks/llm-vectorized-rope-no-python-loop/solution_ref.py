import numpy as np

def apply_rope(x: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Apply Rotary Position Embedding (RoPE) to the last dimension of x."""
    B, S, H, D = x.shape
    # frequency for each pair
    freqs = 1.0 / (10000.0 ** (np.arange(0, D, 2, dtype=np.float64) / D))
    # angles = pos * freqs, shape (S, D//2)
    angles = pos[:, None].astype(np.float64) * freqs[None, :]
    cos_vals = np.cos(angles)
    sin_vals = np.sin(angles)

    # reshape x into pairs (..., D//2, 2)
    x_pairs = x.reshape(B, S, H, D // 2, 2)
    x_even = x_pairs[..., 0]
    x_odd = x_pairs[..., 1]

    # broadcast cos/sin
    cos_b = cos_vals[None, :, None, :]  # (1, S, 1, D//2)
    sin_b = sin_vals[None, :, None, :]

    # rotate
    y_even = x_even * cos_b - x_odd * sin_b
    y_odd = x_odd * cos_b + x_even * sin_b

    # interleave
    y_pairs = np.stack([y_even, y_odd], axis=-1)
    return y_pairs.reshape(B, S, H, D)
