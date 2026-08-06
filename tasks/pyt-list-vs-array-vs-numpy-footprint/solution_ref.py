import sys
from array import array


def footprint_ratios(n):
    values = list(range(n))

    list_bytes = sys.getsizeof(values)

    arr = array("q", values)
    array_bytes = sys.getsizeof(arr)

    return {
        "list_vs_array": list_bytes / array_bytes,
    }
