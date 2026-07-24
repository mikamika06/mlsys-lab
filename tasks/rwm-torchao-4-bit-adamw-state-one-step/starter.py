import numpy as np


def adamw_4bit_step(
    p: np.ndarray, grad: np.ndarray,
    m_packed: np.ndarray, m_scales: np.ndarray,
    v_packed: np.ndarray, v_scales: np.ndarray,
    step: int, block_size: int = 32,
    lr: float = 1e-3, beta1: float = 0.9, beta2: float = 0.999,
    eps: float = 1e-8, weight_decay: float = 0.01,
) -> dict:
    """
    Dequantize m,v (4-bit blockwise), take one AdamW step, requantize m,v.

    Returns {"p_new", "m_packed", "m_scales", "v_packed", "v_scales"}.
    """
    raise NotImplementedError('your code here')
