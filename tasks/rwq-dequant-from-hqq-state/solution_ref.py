import numpy as np

def dequant_from_hqq_state(W_q: np.ndarray, scale: np.ndarray, zero: np.ndarray) -> np.ndarray:
    """
    Dequantize a HQQ weight matrix.

    Parameters
    ----------
    W_q : np.ndarray
        Integer codes of shape (n, m).
    scale : np.ndarray
        Per‑column scales of shape (m,).
    zero : np.ndarray
        Per‑column zero points of shape (m,).

    Returns
    -------
    np.ndarray
        Dequantized weights as float64.
    """
    W_q_f = np.asarray(W_q, dtype=np.float64)
    scale_f = np.asarray(scale, dtype=np.float64)
    zero_f  = np.asarray(zero, dtype=np.float64)
    return (W_q_f - zero_f) * scale_f
