import numpy as np


def layernorm_fp16_welford(x, gamma, beta, eps=1e-5) -> np.ndarray:
    """LayerNorm of the length-n vector x, computed entirely in float16 with the
    variance obtained from Welford's online (single-pass) algorithm.

    x, gamma, beta: length-n arrays. Returns a length-n float16 array.
    Cast inputs to np.float16 and keep every intermediate in float16, accumulating
    strictly left-to-right (see task.md for the exact procedure).
    """
    raise NotImplementedError("your code here")


def layernorm_fp16_two_pass(x, gamma, beta, eps=1e-5) -> np.ndarray:
    """LayerNorm of the length-n vector x, computed entirely in float16 with the
    variance obtained from the two-pass algorithm (running sum -> mean, then a
    second pass over squared deviations).

    x, gamma, beta: length-n arrays. Returns a length-n float16 array.
    Cast inputs to np.float16 and keep every intermediate in float16, accumulating
    strictly left-to-right (see task.md for the exact procedure).
    """
    raise NotImplementedError("your code here")
