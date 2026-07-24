import numpy as np

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
    max_abs = np.max(np.abs(W))
    if max_abs == 0:
        scale = 1.0
    else:
        scale = max_abs / 127.0
    q = np.round(W / scale).astype(np.int8)
    dq = q.astype(np.float64) * scale
    return q, dq
