import numpy as np


def detect_truncation(imatrix_data, expected_tensors):
    keys = list(imatrix_data.keys())
    missing_count = max(0, expected_tensors - len(keys))
    zero_blocks = 0
    for k, v in imatrix_data.items():
        arr = np.array(v, dtype=np.float64)
        if arr.size > 0 and np.all(arr == 0.0):
            zero_blocks += 1
    is_truncated = (missing_count > 0) or (zero_blocks > 0)
    return {
        "is_truncated": is_truncated,
        "missing_tensors": missing_count,
        "zero_blocks": zero_blocks,
    }
