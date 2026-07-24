import numpy as np

def dynamic_activation_scale_zero_point(x: np.ndarray):
    """
    Compute the dynamic quantization parameters for a batch of activations.

    Parameters
    ----------
    x : np.ndarray
        1‑D array of floating‑point activation values.

    Returns
    -------
    scale : float
        The scaling factor used to map real values to uint8.
    zero_point : int
        Integer in [0,255] such that the value 0 maps to this index.
    """
    min_val = np.min(x)
    max_val = np.max(x)

    if max_val == min_val:
        scale = 1.0
        zero_point = 128
    else:
        scale = (max_val - min_val) / 255.0
        zero_point = int(round(-min_val / scale))
        zero_point = np.clip(zero_point, 0, 255)

    return float(scale), int(zero_point)
