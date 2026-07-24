import sys

import numpy as np


def list_footprint_ratio(values: list) -> float:
    """(list container + distinct boxed-int objects) / numpy int64 array footprint."""
    list_bytes = sys.getsizeof(values)

    seen = {}
    for v in values:
        seen[id(v)] = v
    list_bytes += sum(sys.getsizeof(v) for v in seen.values())

    arr = np.array(values, dtype=np.int64)
    array_bytes = sys.getsizeof(arr)

    return list_bytes / array_bytes if array_bytes else 0.0
