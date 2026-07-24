import numpy as np

# The fixed NF4 codebook (bitsandbytes QLoRA), 16 quantile-derived levels in [-1, 1].
_NF4 = np.array([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
    0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
    0.7229568362236023, 1.0,
], dtype=np.float64)


def nf4_vs_int4_mse(w: np.ndarray):
    """
    Quantize a block of near-Gaussian weights two ways and return their
    reconstruction MSEs as (mse_nf4, mse_int4):

    - NF4: normalize by absmax, snap each value to the nearest of the 16
      fixed NF4 codebook levels, scale back by absmax.
    - Symmetric INT4: scale = absmax / 7, round(w / scale) clipped to
      [-8, 7], dequantize by * scale.
    """
    w = np.asarray(w, dtype=np.float64)
    absmax = float(np.max(np.abs(w))) or 1.0

    wn = w / absmax
    d = np.abs(wn[:, None] - _NF4[None, :])
    idx = np.argmin(d, axis=1)
    deq_nf4 = _NF4[idx] * absmax
    mse_nf4 = float(np.mean((w - deq_nf4) ** 2))

    scale = absmax / 7.0
    codes = np.clip(np.round(w / scale), -8, 7)
    deq_int4 = codes * scale
    mse_int4 = float(np.mean((w - deq_int4) ** 2))

    return mse_nf4, mse_int4
