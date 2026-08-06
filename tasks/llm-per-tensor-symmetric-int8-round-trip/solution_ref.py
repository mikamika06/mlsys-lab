import numpy as np
import math

def per_tensor_int8_round_trip(W: np.ndarray):
    """
    Symmetric per‑tensor int8 quantization and dequantization.

    Parameters
    ----------
    W : np.ndarray
        Input tensor of arbitrary shape.

    Returns
    -------
    q : np.ndarray[np.int8]
        Quantized integer codes.
    dq : np.ndarray[np.float64]
        Dequantized double‑precision values.
    """
    max_abs = 0.0
    for x in W.flat:
        val = x if x >= 0.0 else -x
        if val > max_abs:
            max_abs = val

    if max_abs == 0:
        scale = 1.0
    else:
        scale = max_abs / 127.0

    q = np.empty(W.shape, dtype=np.int8)
    dq = np.empty(W.shape, dtype=np.float64)

    flat_W = W.flat
    flat_q = q.flat
    flat_dq = dq.flat

    for i in range(W.size):
        val = flat_W[i] / scale
        if val >= 0.0:
            rounded = math.floor(val + 0.5)
        else:
            rounded = math.ceil(val - 0.5)
        
        if rounded > 127:
            rounded = 127
        elif rounded < -128:
            rounded = -128

        qi = int(rounded)
        flat_q[i] = qi
        flat_dq[i] = float(qi) * scale

    return q, dq
