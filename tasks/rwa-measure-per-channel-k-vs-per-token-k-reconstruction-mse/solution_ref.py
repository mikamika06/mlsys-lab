import numpy as np


def _affine_quant_dequant(x: np.ndarray, bits: int, axis) -> np.ndarray:
    """Uniform affine (asymmetric) min-max quantizer/dequantizer.

    The min/max (and therefore the scale/zero-point) are computed
    per-group along `axis`; every element in a group shares that one
    scale/zero-point.
    """
    x = np.asarray(x, dtype=np.float64)
    qmax = (1 << bits) - 1
    
    rows, cols = x.shape
    out = np.empty((rows, cols), dtype=np.float64)
    
    if axis == 0:
        xmin = np.empty((1, cols), dtype=np.float64)
        xmax = np.empty((1, cols), dtype=np.float64)
        for c in range(cols):
            mn = x[0, c]
            mx = x[0, c]
            for r in range(1, rows):
                val = x[r, c]
                if val < mn:
                    mn = val
                if val > mx:
                    mx = val
            xmin[0, c] = mn
            xmax[0, c] = mx
            
        scale = np.empty((1, cols), dtype=np.float64)
        for c in range(cols):
            s = (xmax[0, c] - xmin[0, c]) / qmax
            if s == 0.0:
                s = 1.0
            scale[0, c] = s
            
        zero_point = np.empty((1, cols), dtype=np.float64)
        for c in range(cols):
            zero_point[0, c] = round(-xmin[0, c] / scale[0, c])
            
        for c in range(cols):
            s = scale[0, c]
            zp = zero_point[0, c]
            for r in range(rows):
                q = round(x[r, c] / s + zp)
                if q < 0:
                    q = 0.0
                elif q > qmax:
                    q = float(qmax)
                else:
                    q = float(q)
                out[r, c] = (q - zp) * s
                
    elif axis == 1:
        xmin = np.empty((rows, 1), dtype=np.float64)
        xmax = np.empty((rows, 1), dtype=np.float64)
        for r in range(rows):
            mn = x[r, 0]
            mx = x[r, 0]
            for c in range(1, cols):
                val = x[r, c]
                if val < mn:
                    mn = val
                if val > mx:
                    mx = val
            xmin[r, 0] = mn
            xmax[r, 0] = mx
            
        scale = np.empty((rows, 1), dtype=np.float64)
        for r in range(rows):
            s = (xmax[r, 0] - xmin[r, 0]) / qmax
            if s == 0.0:
                s = 1.0
            scale[r, 0] = s
            
        zero_point = np.empty((rows, 1), dtype=np.float64)
        for r in range(rows):
            zero_point[r, 0] = round(-xmin[r, 0] / scale[r, 0])
            
        for r in range(rows):
            s = scale[r, 0]
            zp = zero_point[r, 0]
            for c in range(cols):
                q = round(x[r, c] / s + zp)
                if q < 0:
                    q = 0.0
                elif q > qmax:
                    q = float(qmax)
                else:
                    q = float(q)
                out[r, c] = (q - zp) * s
                
    return out


def per_channel_vs_per_token_k_mse(K: np.ndarray, bits: int) -> np.ndarray:
    """
    K: (n_tokens, d_channels) fp64 key cache. bits: quantizer bit-width.

    Quantize K two ways with a uniform affine min-max quantizer:
      - per-channel: one scale/zero-point per COLUMN, min/max taken across
        all tokens (axis=0).
      - per-token: one scale/zero-point per ROW, min/max taken across all
        channels (axis=1).

    Returns np.array([mse_per_channel, mse_per_token]), the reconstruction
    MSE (mean squared error of dequant(quant(K)) vs K) of each scheme.
    """
    K = np.asarray(K, dtype=np.float64)

    K_per_channel = _affine_quant_dequant(K, bits, axis=0)
    K_per_token = _affine_quant_dequant(K, bits, axis=1)

    rows, cols = K.shape
    total_elements = rows * cols

    sum_sq_channel = 0.0
    for r in range(rows):
        for c in range(cols):
            diff = K_per_channel[r, c] - K[r, c]
            sum_sq_channel += diff * diff
    mse_per_channel = float(sum_sq_channel / total_elements)

    sum_sq_token = 0.0
    for r in range(rows):
        for c in range(cols):
            diff = K_per_token[r, c] - K[r, c]
            sum_sq_token += diff * diff
    mse_per_token = float(sum_sq_token / total_elements)

    return np.array([mse_per_channel, mse_per_token])
