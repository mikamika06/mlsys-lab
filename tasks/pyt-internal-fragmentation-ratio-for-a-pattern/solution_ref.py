import sys


def internal_fragmentation_ratio(pattern):
    values = list(pattern)
    requested = sum(values)
    if requested == 0:
        raise ValueError("empty allocation request")
    resident = 0
    objects = []
    for size in values:
        obj = bytearray(size)
        objects.append(obj)
        resident += sys.getsizeof(obj)
    objects.clear()
    return float(resident / requested)
