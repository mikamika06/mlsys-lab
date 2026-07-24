import sys
from array import array
import numpy as np


def footprint_ratios(n):
    values = list(range(n))

    list_bytes = sys.getsizeof(values)

    arr = array("q", values)
    array_bytes = sys.getsizeof(arr)

    np_arr = np.asarray(values, dtype=np.int64)
    numpy_bytes = np_arr.nbytes

    return {
        "list_vs_array": list_bytes / array_bytes,
        "list_vs_numpy": list_bytes / numpy_bytes,
        "array_vs_numpy": array_bytes / numpy_bytes,
    }
