import numpy as np

def integer_conv2d(i_q: np.ndarray, i_z: int, w_q: np.ndarray, b_q: np.ndarray) -> np.ndarray:
    """
    i_q: shape (1, H, W, Cin), dtype uint8 or int8
    i_z: int, input zero point
    w_q: shape (Cout, Kh, Kw, Cin), dtype int8
    b_q: shape (Cout,), dtype int32
    
    Returns:
    int32 array of shape (1, H_out, W_out, Cout)
    Padding is valid, stride is 1.
    """
    raise NotImplementedError
