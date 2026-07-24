import numpy as np


def fp8_dynamic_matmul(W: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    FP8 E4M3 W8A8 "dynamic" quantized matmul: Y ~= W @ X.

    - `W` (M, K): weights, quantized with a single per-tensor scale
      (scale_w = max(|W|) / 448).
    - `X` (K, N): activations, quantized with a per-token scale, i.e. one
      scale per column of `X` computed on the fly from `X` itself
      (scale_x[j] = max(|X[:, j]|) / 448).

    Cast both `W / scale_w` and `X / scale_x` to the nearest representable
    E4M3 value (4 exponent bits, 3 mantissa bits, bias 7, max magnitude
    448, clamp anything larger), matmul the quantized operands, then
    dequantize by multiplying back by scale_w * scale_x[token].

    Returns Y as an array of shape (M, N).
    """
    raise NotImplementedError('your code here')
