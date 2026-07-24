import numpy as np


def int4_groupwise_quant(W: np.ndarray, group_size: int = 32):
    """
    W: (rows, cols) weight matrix; `cols` must be a multiple of
        `group_size`.

    Symmetric int4 quantization, applied independently per row and per
    contiguous group of `group_size` values along the columns:

        amax  = max(abs(group))
        scale = amax / 8              (1.0 if amax == 0, to avoid /0)
        code  = clip(round(group / scale), -8, 7)

    Returns (codes, scales):
      codes: (rows, cols) int array, values in [-8, 7].
      scales: (rows, cols // group_size) float array, one scale per row
        per group.
    """
    W = np.asarray(W, dtype=np.float64)
    rows, cols = W.shape
    n_groups = cols // group_size
    Wg = W.reshape(rows, n_groups, group_size)

    amax = np.max(np.abs(Wg), axis=-1)
    scales = np.where(amax == 0, 1.0, amax / 8.0)

    codes_g = np.clip(np.round(Wg / scales[:, :, None]), -8, 7).astype(np.int64)
    codes = codes_g.reshape(rows, cols)
    return codes, scales
