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
    _, H, W, Cin = i_q.shape
    Cout, Kh, Kw, _ = w_q.shape
    
    H_out = H - Kh + 1
    W_out = W - Kw + 1
    
    out = np.zeros((1, H_out, W_out, Cout), dtype=np.int32)
    
    i_shifted = i_q.astype(np.int32) - i_z
    w_int = w_q.astype(np.int32)
    
    for y in range(H_out):
        for x in range(W_out):
            patch = i_shifted[:, y:y+Kh, x:x+Kw, :]
            out[:, y, x, :] = np.tensordot(patch, w_int, axes=([1, 2, 3], [1, 2, 3])) + b_q
            
    return out
