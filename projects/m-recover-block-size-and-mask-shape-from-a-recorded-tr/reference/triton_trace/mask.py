import numpy as np


def recover_mask_shape(trace_events):
    masks = [e.get("mask") for e in trace_events if "mask" in e]
    if not masks:
        return (0, 0)
    arr = np.array(masks, dtype=bool)
    if arr.ndim == 1:
        valid_indices = np.where(arr)[0]
        if len(valid_indices) == 0:
            return (0,)
        return (int(valid_indices[-1] - valid_indices[0] + 1),)
    sum_cols = np.sum(arr, axis=0)
    valid_cols = np.where(sum_cols > 0)[0]
    sum_rows = np.sum(arr, axis=1)
    valid_rows = np.where(sum_rows > 0)[0]
    h = int(valid_rows[-1] - valid_rows[0] + 1) if len(valid_rows) > 0 else 0
    w = int(valid_cols[-1] - valid_cols[0] + 1) if len(valid_cols) > 0 else 0
    return (h, w)
