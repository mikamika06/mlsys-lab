import numpy as np


def nhwc_to_nchw(arr: np.ndarray) -> np.ndarray:
    """Converts an NHWC tensor to NCHW contiguous layout."""
    return np.ascontiguousarray(np.transpose(arr, (0, 3, 1, 2)))


def nchw_to_nhwc(arr: np.ndarray) -> np.ndarray:
    """Converts an NCHW tensor to NHWC contiguous layout."""
    return np.ascontiguousarray(np.transpose(arr, (0, 2, 3, 1)))


def measure_transpose_bytes(shape: tuple, dtype: np.dtype) -> dict:
    """Calculates memory requirements for layout conversion."""
    dt = np.dtype(dtype)
    num_elements = int(np.prod(shape))
    total_bytes = num_elements * dt.itemsize
    return {
        "num_elements": num_elements,
        "bytes": total_bytes,
        "intermediate_allocations": 2 * total_bytes,
    }
