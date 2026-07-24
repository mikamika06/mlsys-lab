import numpy as np


def quantize_dequantize_v_blockwise(v: np.ndarray, blocksize: int):
    """
    Build the 256-entry bitsandbytes-style dynamic exponent map, quantize
    `v` blockwise (absmax-normalize per block, snap to nearest map code),
    dequantize, and return (v_hat, codes, absmax) as described in task.md.
    """
    raise NotImplementedError('your code here')
