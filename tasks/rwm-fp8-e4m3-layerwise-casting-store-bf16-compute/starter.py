import numpy as np


def cast_and_matmul_fp8e4m3(W: np.ndarray, X: np.ndarray):
    """
    Encode W to E4M3 storage codes (uint8), decode + round to bfloat16 for
    compute, round X to bfloat16, matmul with float32 accumulation.
    Return (Y, codes) as described in task.md.
    """
    raise NotImplementedError('your code here')
