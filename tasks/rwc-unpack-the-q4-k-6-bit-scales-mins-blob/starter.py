import numpy as np


def unpack_q4k_scales_mins(packed: np.ndarray):
    """Unpack the 12-byte Q4_K scales/mins blob into 8 scales + 8 mins.

    Parameters
    ----------
    packed : np.ndarray, shape (12,), uint8
        The packed scales/mins blob.

    Returns
    -------
    (scales, mins) : tuple[np.ndarray, np.ndarray]
        Each shape (8,), dtype uint8, values in [0, 63].
    """
    raise NotImplementedError('your code here')
