import numpy as np


def quantize_classifier_head(X, W, b):
    """Per-channel int8 quantization of a classifier head, then forward pass.

    X: (N, D) input activations. W: (C, D) weight matrix. b: (C,) bias.
    Quantize each row of W independently to int8 (symmetric, scale =
    max(|row|)/127), dequantize, and compute logits = X @ W_deq.T + b.
    Returns (logits, W_int8, scale):
      logits: (N, C) float array.
      W_int8: (C, D) integer array in [-127, 127].
      scale: (C,) float array of per-channel scales.
    """
    raise NotImplementedError('your code here')
