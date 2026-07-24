import numpy as np


def _sym_int4_dequant(x: np.ndarray, scale: np.ndarray) -> np.ndarray:
    scale_safe = np.where(scale == 0, 1.0, scale)
    code = np.clip(np.round(x / scale_safe), -7, 7)
    return code * scale_safe


def qint4_granularity_mse(W: np.ndarray, group_size: int = 32):
    W = np.asarray(W, dtype=np.float64)
    rows, cols = W.shape

    # per-axis: one symmetric int4 scale per row
    amax_axis = np.max(np.abs(W), axis=1)
    scale_axis = amax_axis / 7.0
    deq_axis = _sym_int4_dequant(W, scale_axis[:, None])
    mse_per_axis = float(np.mean((W - deq_axis) ** 2))

    # per-group: one symmetric int4 scale per group_size-column group
    ng = cols // group_size
    Wg = W.reshape(rows, ng, group_size)
    amax_group = np.max(np.abs(Wg), axis=2)
    scale_group = amax_group / 7.0
    deq_group = _sym_int4_dequant(Wg, scale_group[:, :, None]).reshape(rows, cols)
    mse_per_group = float(np.mean((W - deq_group) ** 2))

    return mse_per_axis, mse_per_group
