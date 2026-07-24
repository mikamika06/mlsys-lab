import numpy as np

def fused_rope_qk(
    Q: np.ndarray,
    K: np.ndarray,
    sin: np.ndarray,
    cos: np.ndarray
) -> np.ndarray:
    """
    Explicitly rotate queries and keys with RoPE and compute the dot product.
    This reference implementation is fully vectorised and serves as the oracle
    for grading.  It does *not* fuse the rotation into the score computation;
    it materialises rotated tensors first, which is correct but less efficient.
    """
    Q_even = Q[..., ::2]
    Q_odd  = Q[..., 1::2]
    K_even = K[..., ::2]
    K_odd  = K[..., 1::2]

    Q_rot_even = Q_even * cos - Q_odd * sin
    Q_rot_odd  = Q_even * sin + Q_odd * cos
    K_rot_even = K_even * cos - K_odd * sin
    K_rot_odd  = K_even * sin + K_odd * cos

    Q_rot = np.concatenate([Q_rot_even, Q_rot_odd], axis=-1)
    K_rot = np.concatenate([K_rot_even, K_rot_odd], axis=-1)

    return np.einsum('bld,bmd->blm', Q_rot, K_rot)
