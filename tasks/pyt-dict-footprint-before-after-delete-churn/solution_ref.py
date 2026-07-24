import sys


def dict_footprint_churn(n: int, cycles: int) -> tuple[int, int]:
    d = {i: i for i in range(n)}
    before = sys.getsizeof(d)

    next_key = n
    for _ in range(cycles):
        for k in list(d):
            del d[k]
        for _ in range(n):
            d[next_key] = next_key
            next_key += 1

    after = sys.getsizeof(d)
    return before, after
