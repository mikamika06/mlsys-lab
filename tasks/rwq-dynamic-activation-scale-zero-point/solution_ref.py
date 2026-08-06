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
    min_val = x[0]
    max_val = x[0]
    for i in range(1, len(x)):
        val = x[i]
        if val < min_val:
            min_val = val
        if val > max_val:
            max_val = val

    if max_val == min_val:
        scale = 1.0
        zero_point = 128
    else:
        scale = (max_val - min_val) / 255.0
        zero_point = int(round(-min_val / scale))
        if zero_point < 0:
            zero_point = 0
        elif zero_point > 255:
            zero_point = 255

    return float(scale), int(zero_point)
