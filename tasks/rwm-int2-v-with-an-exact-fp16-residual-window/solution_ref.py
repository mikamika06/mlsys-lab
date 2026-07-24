import numpy as np


def kv_int2_residual_window(V: np.ndarray, group_size: int = 32, residual_window: int = 16) -> np.ndarray:
    """
    Quantize all but the last `residual_window` rows of `V` to 2 bits/element
    using grouped affine (zero-point) quantization along the channel axis;
    leave the last `residual_window` rows exact. Returns the reconstructed
    (T, d) array.
    """
    V = np.asarray(V, dtype=np.float64)
    T, d = V.shape
    Tq = T - residual_window

    Vq = V[:Tq]
    Vr = V[Tq:]

    ng = d // group_size
    Vq_g = Vq.reshape(Tq, ng, group_size)

    lo = np.min(Vq_g, axis=-1)
    hi = np.max(Vq_g, axis=-1)
    scale = (hi - lo) / 3.0
    scale = np.where(scale == 0, 1.0, scale)

    code = np.round((Vq_g - lo[:, :, None]) / scale[:, :, None])
    code = np.clip(code, 0, 3)

    Vq_hat = (code * scale[:, :, None] + lo[:, :, None]).reshape(Tq, d)

    return np.concatenate([Vq_hat, Vr], axis=0)
