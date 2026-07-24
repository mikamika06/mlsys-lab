import numpy as np

def alibi_slopes(n_heads: int) -> np.ndarray:
    """
    Return the ALiBi slopes for `n_heads` attention heads.

    Parameters
    ----------
    n_heads : int
        Number of attention heads. Must be a positive integer.

    Returns
    -------
    slopes : np.ndarray, dtype=np.float32
        Slopes in increasing order of head index.
    """
    if not isinstance(n_heads, int) or n_heads <= 0:
        raise ValueError("n_heads must be a positive integer")

    def _get_power_of_two_slopes(k: int):
        start = 2 ** (-(k + 3))
        ratio = 2.0
        return np.array([start * (ratio ** i) for i in range(k)], dtype=np.float32)

    if n_heads & (n_heads - 1) == 0:
        # exact power of two
        return _get_power_of_two_slopes(n_heads)
    else:
        k = 2 ** int(np.floor(np.log2(n_heads)))
        slopes = _get_power_of_two_slopes(k).tolist()
        extra = alibi_slopes(n_heads - k).tolist()
        slopes.extend(extra)
        return np.array(slopes, dtype=np.float32)
