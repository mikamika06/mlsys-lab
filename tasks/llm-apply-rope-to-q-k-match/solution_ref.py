import numpy as np

def apply_rope(x: np.ndarray, pos: int) -> np.ndarray:
    """
    Apply Rotary Position Embedding to a batch of vectors.

    Parameters
    ----------
    x : np.ndarray
        Input array of shape (n, d) with even d.
    pos : int
        Token position index used to scale the frequency vector.

    Returns
    -------
    np.ndarray
        Rotated array of the same shape and dtype float64.
    """
    x = np.asarray(x, dtype=np.float64)
    n, d = x.shape
    if d % 2 != 0:
        raise ValueError("Dimension must be even for RoPE.")
    omega = np.linspace(0.01, 0.99, d // 2)
    theta = pos * omega
    cos = np.cos(theta)
    sin = np.sin(theta)
    even = x[:, ::2]
    odd = x[:, 1::2]
    new_even = even * cos - odd * sin
    new_odd = even * sin + odd * cos
    out = np.empty_like(x)
    out[:, ::2] = new_even
    out[:, 1::2] = new_odd
    return out
