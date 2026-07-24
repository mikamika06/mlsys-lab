import numpy as np


def _group_quant_dequant(x: np.ndarray, bits: int) -> np.ndarray:
    """Uniform affine min-max quantizer/dequantizer along the LAST axis:
    every group of `group_size` values along that axis shares one
    scale/zero-point."""
    x = np.asarray(x, dtype=np.float64)
    qmax = (1 << bits) - 1
    xmin = np.min(x, axis=-1, keepdims=True)
    xmax = np.max(x, axis=-1, keepdims=True)
    scale = (xmax - xmin) / qmax
    scale = np.where(scale == 0, 1.0, scale)
    zero_point = np.round(-xmin / scale)
    q = np.clip(np.round(x / scale + zero_point), 0, qmax)
    return (q - zero_point) * scale


def awq_apply_fixed_scale(
    W: np.ndarray, s: np.ndarray, X: np.ndarray, group_size: int, bits: int = 4
) -> np.ndarray:
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

    Steps 1 and 3 cancel exactly in real arithmetic; the whole point is
    that step 2's rounding error is smaller after up-scaling channels
    that would otherwise be crushed by outlier-dominated group ranges.

    Returns `output`, shape (batch, out_features).
    """
    W = np.asarray(W, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    out_features, in_features = W.shape

    Ws = W * s[None, :]
    Ws_grouped = Ws.reshape(out_features, in_features // group_size, group_size)
    Wq_grouped = _group_quant_dequant(Ws_grouped, bits)
    Wq = Wq_grouped.reshape(out_features, in_features)

    Xs = X / s[None, :]
    return Xs @ Wq.T
