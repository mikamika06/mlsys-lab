import numpy as np


def lora_delta_forward(x, base, A, B, scale):
    """
    x: (n, d), base: (n, d) frozen base layer output for x, A: (d, r),
    B: (r, d), scale: float.

    Returns base + scale * (x @ A) @ B as a float64 (n, d) array.
    """
    raise NotImplementedError('your code here')
