import numpy as np


def awq_scale_and_quantize(W: np.ndarray, X: np.ndarray, s: np.ndarray, group_size: int, bits: int = 4):
    """AWQ-style per-channel scale migration + int-`bits` group quant.

    W: (out_features, in_features) float64 -- a Linear layer's weight.
    X: (batch, in_features) float64 -- activations feeding that layer.
    s: (in_features,) float64, positive per-input-channel scale
        (AWQ's job elsewhere is choosing `s` to protect salient
        channels; here it's given).
    group_size: positive int dividing in_features.
    bits: bit width for the weight quantizer (default 4).

    1. Migrate the scale: W' = W * s (broadcast over columns, i.e. per
       input channel), X' = X / s (same per-channel scale, inverted).
       Because the scale is applied to one side and its inverse to the
       other, X' @ W'^T == X @ W^T exactly (up to floating point) --
       this identity is what lets AWQ shrink salient weight channels
       (making them easier to quantize) without changing the layer's
       output.
    2. Quantize W' with an int-`bits` group-affine quantizer, PER
       OUTPUT ROW, grouped along the in_features axis in contiguous
       chunks of `group_size` (each group gets its own scale/zero
       point, same formulas as standard int-n group quant):
           qmax  = 2**bits - 1
           scale = (max(g) - min(g)) / qmax
           zero  = clip(round(-min(g) / scale), 0, qmax)
           code  = clip(round(g / scale) + zero, 0, qmax)
           g_hat = (code - zero) * scale
       (a constant group reconstructs itself exactly) -- call the
       result W_hat, same shape as W'.

    Returns (Y_identity, Y_quant):
      Y_identity = X' @ W'^T           (should equal X @ W^T)
      Y_quant    = X' @ W_hat^T        (the actual quantized output)
    """
    raise NotImplementedError('your code here')
