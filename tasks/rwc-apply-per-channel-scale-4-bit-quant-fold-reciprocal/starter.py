import math

def awq_apply_fixed_scale(W: list[list[float]], s: list[float], X: list[list[float]], group_size: int, bits: int=4) -> list[list[float]]:
    """
    W: (out_features, in_features) weight matrix.
    s: (in_features,) positive per-input-channel AWQ smoothing scale
        (already chosen -- this function just applies it).
    X: (batch, in_features) activations.
    group_size: number of consecutive input channels sharing one
        quantizer scale/zero-point (in_features must be a multiple of
        group_size).
    bits: quantizer bit-width.

    AWQ transform for a fixed scale:
      1. Ws = W * s (scale weight COLUMNS, i.e. input channels, up by s)
      2. Wq = fake-quantize Ws to `bits` bits, GROUP-WISE along the
         input-channel axis: for each output row and each contiguous
         block of `group_size` input channels, use a uniform affine
         min-max quantizer/dequantizer over just that block.
      3. Xs = X / s (fold the reciprocal scale into activations)
      4. output = Xs @ Wq.T

    Returns `output`, shape (batch, out_features).
    """
    raise NotImplementedError('your code here')
