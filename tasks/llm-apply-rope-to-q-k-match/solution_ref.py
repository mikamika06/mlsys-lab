import math


def apply_rope(x: list[list[float]], pos: int) -> list[list[float]]:
    """
    Apply Rotary Position Embedding to a batch of vectors.

    Parameters
    ----------
    x : list[list[float]]
        Input list of shape (n, d) with even d.
    pos : int
        Token position index used to scale the frequency vector.

    Returns
    -------
    list[list[float]]
        Rotated list of the same shape.
    """
    n = len(x)
    if n == 0:
        return []
    d = len(x[0])
    if d % 2 != 0:
        raise ValueError("Dimension must be even for RoPE.")

    out = [[0.0] * d for _ in range(n)]
    half_d = d // 2

    for j in range(half_d):
        if half_d == 1:
            omega_j = 0.01
        else:
            omega_j = 0.01 + j * (0.99 - 0.01) / (half_d - 1)
        theta_j = pos * omega_j
        cos_j = math.cos(theta_j)
        sin_j = math.sin(theta_j)

        even_idx = 2 * j
        odd_idx = 2 * j + 1

        for i in range(n):
            even_val = x[i][even_idx]
            odd_val = x[i][odd_idx]
            out[i][even_idx] = even_val * cos_j - odd_val * sin_j
            out[i][odd_idx] = even_val * sin_j + odd_val * cos_j

    return out
