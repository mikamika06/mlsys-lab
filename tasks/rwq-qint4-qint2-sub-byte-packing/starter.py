import numpy as np


def pack_sub_byte(W: np.ndarray, nbits: int):
    """
    Per-row symmetric quantization (scale = max(|row|) / (2^(nbits-1)-1)),
    then pack `8 // nbits` low-bit unsigned codes into each uint8 byte,
    least-significant code first, as described in task.md.

    Returns (packed, s, dequant):
      packed  -- uint8 array, shape (d_out, d_in * nbits // 8).
      s       -- float array, shape (d_out,), per-row scale.
      dequant -- float array, shape (d_out, d_in), reconstruction obtained
                 by unpacking `packed` and mapping back to float.
    """
    raise NotImplementedError('your code here')
