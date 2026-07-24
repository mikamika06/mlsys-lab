import numpy as np


def rtn_group_quantize(W: np.ndarray, group_size: int):
    """
    Per-row, per-group symmetric int4 round-to-nearest quantization, no
    error feedback. `d_in` is guaranteed divisible by `group_size`.

    For each row and each contiguous block of `group_size` columns:
      amax  = max(|W[row, block]|)
      scale = amax / 7   (or 1.0 if amax == 0)
      code  = clip(round(w / scale), -7, 7)
    Dequantized reconstruction is code * scale.

    Returns (codes, Wq):
      codes -- integer array, same shape as W, values in [-7, 7].
      Wq    -- float array, same shape as W, the dequantized reconstruction.
    """
    W = np.asarray(W, dtype=np.float64)
    qmax = 7
    d_out, d_in = W.shape
    n_groups = d_in // group_size

    codes = np.empty((d_out, d_in), dtype=np.int64)
    Wq = np.empty((d_out, d_in), dtype=np.float64)
    for g in range(n_groups):
        sl = slice(g * group_size, (g + 1) * group_size)
        block = W[:, sl]
        amax = np.max(np.abs(block), axis=1)
        scale = np.where(amax > 0, amax / qmax, 1.0)
        c = np.clip(np.round(block / scale[:, None]), -qmax, qmax)
        codes[:, sl] = c.astype(np.int64)
        Wq[:, sl] = c * scale[:, None]
    return codes, Wq
