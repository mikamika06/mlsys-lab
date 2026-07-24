import numpy as np


def rtn_group_quantize(W: np.ndarray, group_size: int):
    """
    Per-row, per-group symmetric int4 round-to-nearest quantization (no
    error feedback), as described in task.md. `d_in` is guaranteed
    divisible by `group_size`.

    Returns (codes, Wq):
      codes -- integer array, same shape as W, values in [-7, 7].
      Wq    -- float array, same shape as W, the dequantized reconstruction
               (codes * per-row-per-group scale).
    """
    raise NotImplementedError('your code here')
