import sys


def list_footprint_ratio(values: list[int]) -> float:
    """(list container + distinct boxed-int objects) / flat array footprint."""
    list_bytes = sys.getsizeof(values)

    seen = {}
    for v in values:
        seen[id(v)] = v
    list_bytes += sum(sys.getsizeof(v) for v in seen.values())

    array_bytes = 112 + len(values) * 8

    return list_bytes / array_bytes if array_bytes else 0.0
