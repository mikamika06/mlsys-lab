import math

def rope_pi(seq_len: int, dim: int, L_train: float, L_new: float) -> tuple[list[list[float]], list[list[float]]]:
    """RoPE with Position Interpolation: scale only the positions."""
    pos = [float(i) for i in range(seq_len)]
    scale = L_train / L_new
    pos_scaled = [p * scale for p in pos]                              # scale positions only
    half_dim = dim // 2
    theta = [1.0 / (10000.0 ** (2.0 * k / dim)) for k in range(half_dim)]          # no extra scale

    cos_matrix = []
    sin_matrix = []
    for p in pos_scaled:
        cos_row = []
        sin_row = []
        for t in theta:
            angle = p * t
            cos_row.append(math.cos(angle))
            sin_row.append(math.sin(angle))
        cos_matrix.append(cos_row)
        sin_matrix.append(sin_row)
    return cos_matrix, sin_matrix
