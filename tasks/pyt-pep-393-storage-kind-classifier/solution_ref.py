import sys


def _kind_from_size(s):
    n = len(s)
    target = sys.getsizeof(s)

    if target == sys.getsizeof("A" * n):
        return 1

    for kind, candidate in (
        (1, "é" * n),
        (2, "Ā" * n),
        (4, "😀" * n),
    ):
        if target == sys.getsizeof(candidate):
            return kind

    raise RuntimeError("unknown CPython unicode storage kind")


def classify_storage_kind(strings):
    return [_kind_from_size(s) for s in strings]
