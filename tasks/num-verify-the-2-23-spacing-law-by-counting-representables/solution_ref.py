import numpy as np

def count_fp32_in_range(start: float, end: float) -> int:
    """
    Return the number of distinct IEEE‑754 single precision values in [start, end).
    Assumes start and end are representable float32 and start < end.
    """
    s = np.float32(start).view(np.uint32)
    e = np.float32(end).view(np.uint32)
    return int(e - s)
