import numpy as np

def scale_rope_base(theta: np.ndarray | float,
                    factor: np.ndarray | float) -> np.ndarray:
    """
    NTK‑aware RoPE base scaling.

    Parameters
    ----------
    theta : array_like or scalar
        Original RoPE base(s).
    factor : array_like or scalar
        Scaling factor(s).  The new base is computed as ``theta ** factor``.
    Returns
    -------
    ndarray
        Scaled base(s) with dtype float64.
    """
    theta_arr = np.asarray(theta, dtype=np.float64)
    factor_arr = np.asarray(factor, dtype=np.float64)
    return theta_arr ** factor_arr
