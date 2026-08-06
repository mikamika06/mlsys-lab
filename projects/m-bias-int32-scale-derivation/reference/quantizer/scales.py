import numpy as np

def derive_bias_scale(input_scale: float, weight_scales: np.ndarray) -> np.ndarray:
    """
    input_scale: float
    weight_scales: 1D array of length Cout
    Returns: 1D float32 array of length Cout
    """
    return (input_scale * weight_scales).astype(np.float32)

def dequantize_weights(w_q: np.ndarray, w_scales: np.ndarray) -> np.ndarray:
    """
    w_q: shape (Cout, Kh, Kw, Cin), dtype int8
    w_scales: shape (Cout,)
    Returns: float32 array of shape (Cout, Kh, Kw, Cin)
    """
    return w_q.astype(np.float32) * w_scales[:, None, None, None]

def quantize_bias(b_real: np.ndarray, b_scales: np.ndarray) -> np.ndarray:
    """
    b_real: shape (Cout,)
    b_scales: shape (Cout,)
    Returns: int32 array
    """
    return np.round(b_real / b_scales).astype(np.int32)
