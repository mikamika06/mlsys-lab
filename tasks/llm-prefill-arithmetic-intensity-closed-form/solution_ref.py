import numpy as np

def prefill_arith_intensity(S: int, d: int, dtype: str = 'float32') -> float:
    """
    Compute the arithmetic intensity of a prefill matrix multiplication
    E (S×d) × W (d×d) → O (S×d).

    Parameters
    ----------
    S : int
        Sequence length.
    d : int
        Hidden dimension.
    dtype : str, optional
        NumPy dtype name; defaults to 'float32'.

    Returns
    -------
    float
        Arithmetic intensity in FLOPs per byte.
    """
    bsize = np.dtype(dtype).itemsize
    return 2 * S * d / ((2 * S + d) * bsize)
