import numpy as np


def dequant_q6_k_superblock(ql: np.ndarray, qh: np.ndarray, scales: np.ndarray, d: float) -> np.ndarray:
    """Dequantize one GGML Q6_K super-block (256 elements) from its packed
    fields: `ql` (128 bytes, low 4 bits x2 per byte), `qh` (64 bytes, high
    2 bits x4 per byte), `scales` (16 signed sub-block scales), and the
    super-block float scale `d`.

    Returns a float64 array of length 256.
    """
    raise NotImplementedError('your code here')
