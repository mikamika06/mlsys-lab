import sys


def reconstruct_capacity(n: int) -> list[int]:
    header = sys.getsizeof([])
    values = []
    result = []
    for i in range(n):
        values.append(i)
        result.append((sys.getsizeof(values) - header) // 8)
    return result
