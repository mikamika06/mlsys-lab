import numpy as np


def qfloat8_weight_quant(W: np.ndarray):
    """Per-tensor scale (to E4M3's max magnitude 448) + nearest-E4M3 cast.

    W: any-shape float64 array.

    1. scale = max(|W|) / 448.0 (use 1.0 if W is all-zero).
    2. W_scaled = W / scale.
    3. Cast every element of W_scaled to the nearest representable
       signed E4M3 value (1 sign, 4 exponent, 3 mantissa; bias 7; no
       infinities; NaN only at exponent==15, mantissa==7): build the
       full grid by enumerating all sign/exponent/mantissa
       combinations and pick the closest value per element.
    4. W_hat = e4m3_values * scale.

    Returns (scale, e4m3_values, W_hat).
    """
    raise NotImplementedError('your code here')
