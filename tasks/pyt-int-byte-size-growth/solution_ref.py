import sys


def int_size_growth():
    values = [
        0,
        1,
        2**30 - 1,
        2**30,
        2**60 - 1,
        2**60,
        2**90,
    ]
    return [sys.getsizeof(value) for value in values]
