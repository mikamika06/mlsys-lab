import math

def rope_pi(seq_len: int, dim: int, L_train: float, L_new: float) -> tuple[list[list[float]], list[list[float]]]:
    """RoPE with Position Interpolation.

    BUG: the scale factor is applied to BOTH the positions AND the frequencies.
    Fix it so scale appears only when computing pos_scaled = pos * scale.

    Returns: (cos, sin) each of shape (seq_len, dim // 2)
    """
    raise NotImplementedError('your code here')
