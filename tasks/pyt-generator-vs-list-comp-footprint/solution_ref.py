import sys


def footprint_ratio(N: int) -> float:
    values = [x for x in range(N)]
    generator = (x for x in range(N))
    return float(sys.getsizeof(values) / sys.getsizeof(generator))
