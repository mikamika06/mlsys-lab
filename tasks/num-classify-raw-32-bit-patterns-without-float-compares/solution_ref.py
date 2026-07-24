import numpy as np

def classify_uint32_patterns(arr: np.ndarray) -> np.ndarray:
    """
    Classify raw 32‑bit IEEE‑754 patterns into five categories while preserving sign.
    Returns an int8 array of labels: s*4 + c, where c is the category index
    (0=Zero,1=Subnormal,2=Normal,3=Infinity,4=NaN).
    """
    # Ensure input type and shape
    arr = np.asarray(arr, dtype=np.uint32)
    sign = (arr >> 31) & 1
    exp = (arr >> 23) & 0xFF
    frac = arr & 0x7FFFFF

    nan_mask = (exp == 255) & (frac != 0)
    inf_mask = (exp == 255) & (frac == 0)
    zero_mask = (exp == 0) & (frac == 0)
    subnormal_mask = (exp == 0) & (frac != 0)
    normal_mask = (~nan_mask & ~inf_mask & ~zero_mask & ~subnormal_mask)

    cat = np.empty_like(arr, dtype=np.int8)
    cat[zero_mask] = 0
    cat[subnormal_mask] = 1
    cat[normal_mask] = 2
    cat[inf_mask] = 3
    cat[nan_mask] = 4

    return sign * 4 + cat
