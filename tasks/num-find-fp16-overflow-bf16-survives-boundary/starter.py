import numpy as np

def find_fp16_overflow_boundary() -> float:
    """TODO: This implementation incorrectly returns the maximum finite FP16 value,
which is not an overflow point.  The correct boundary must be strictly larger
than 65504 and cause FP16 to become inf while BF16 stays finite."""
    raise NotImplementedError('your code here')
