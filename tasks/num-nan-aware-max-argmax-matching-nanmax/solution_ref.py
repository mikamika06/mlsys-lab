import math


def nanmax_argmax(x):
    """Return (max_value, argmax_index) of a 1-D sequence, ignoring NaNs,
    matching np.nanmax/np.nanargmax (first occurrence wins ties). Raises
    ValueError if every element is NaN."""
    best_val = None
    best_idx = -1
    for i, v in enumerate(x):
        v = float(v)
        if math.isnan(v):
            continue
        if best_val is None or v > best_val:
            best_val = v
            best_idx = i
    if best_val is None:
        raise ValueError("all-NaN slice")
    return best_val, best_idx
