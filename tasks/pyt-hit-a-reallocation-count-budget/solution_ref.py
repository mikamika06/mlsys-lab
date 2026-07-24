import sys


def build_list_realloc_count(n: int) -> int:
    sizes = []
    values = [None] * n
    sizes.append(sys.getsizeof(values))
    values[:] = range(n)
    sizes.append(sys.getsizeof(values))
    return sum(1 for a, b in zip(sizes, sizes[1:]) if a != b)
