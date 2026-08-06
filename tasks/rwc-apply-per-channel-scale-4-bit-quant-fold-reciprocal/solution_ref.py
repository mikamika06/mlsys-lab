import math
import numpy as np


def _group_quant_dequant(x: np.ndarray, bits: int) -> np.ndarray:
    """Uniform affine min-max quantizer/dequantizer along the LAST axis:
    every group of `group_size` values along that axis shares one
    scale/zero-point."""
    x = np.asarray(x, dtype=np.float64)
    qmax = (1 << bits) - 1
    shape = x.shape
    out = np.empty(shape, dtype=np.float64)
    
    if len(shape) == 3:
        d0, d1, d2 = shape
        for i in range(d0):
            for j in range(d1):
                xmin = x[i, j, 0]
                xmax = x[i, j, 0]
                for k in range(1, d2):
                    val = x[i, j, k]
                    if val < xmin:
                        xmin = val
                    if val > xmax:
                        xmax = val
                
                scale = (xmax - xmin) / qmax
                if scale == 0:
                    scale = 1.0
                
                zero_point = round(-xmin / scale)
                
                for k in range(d2):
                    q = round(x[i, j, k] / scale + zero_point)
                    if q < 0:
                        q = 0
                    elif q > qmax:
                        q = qmax
                    out[i, j, k] = (q - zero_point) * scale
    elif len(shape) == 2:
        d0, d1 = shape
        for i in range(d0):
            xmin = x[i, 0]
            xmax = x[i, 0]
            for k in range(1, d1):
                val = x[i, k]
                if val < xmin:
                    xmin = val
                if val > xmax:
                    xmax = val
            
            scale = (xmax - xmin) / qmax
            if scale == 0:
                scale = 1.0
            
            zero_point = round(-xmin / scale)
            
            for k in range(d1):
                q = round(x[i, k] / scale + zero_point)
                if q < 0:
                    q = 0
                elif q > qmax:
                    q = qmax
                out[i, k] = (q - zero_point) * scale
    else:
        xmin = x[0]
        xmax = x[0]
        d0 = shape[0]
        for k in range(1, d0):
            val = x[k]
            if val < xmin:
                xmin = val
            if val > xmax:
                xmax = val
        scale = (xmax - xmin) / qmax
        if scale == 0:
            scale = 1.0
        zero_point = round(-xmin / scale)
        for k in range(d0):
            q = round(x[k] / scale + zero_point)
            if q < 0:
                q = 0
            elif q > qmax:
                q = qmax
            out[k] = (q - zero_point) * scale

    return out


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

    Ws = np.empty((out_features, in_features), dtype=np.float64)
    for i in range(out_features):
        for j in range(in_features):
            Ws[i, j] = W[i, j] * s[j]

    num_groups = in_features // group_size
    Ws_grouped = Ws.reshape(out_features, num_groups, group_size)
    Wq_grouped = _group_quant_dequant(Ws_grouped, bits)
    Wq = Wq_grouped.reshape(out_features, in_features)

    batch_size = X.shape[0]
    Xs = np.empty((batch_size, in_features), dtype=np.float64)
    for i in range(batch_size):
        for j in range(in_features):
            Xs[i, j] = X[i, j] / s[j]

    output = np.empty((batch_size, out_features), dtype=np.float64)
    for i in range(batch_size):
        for j in range(out_features):
            acc = 0.0
            for k in range(in_features):
                acc += Xs[i, k] * Wq[j, k]
            output[i, j] = acc

    return output
