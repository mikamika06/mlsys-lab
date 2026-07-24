import sys


class Node:
    # TODO: use __slots__ to remove the per-instance dictionary.
    def __init__(self, value, left, right):
        self.value = value
        self.left = left
        self.right = right


def node_size_ratio(n: int = 100000) -> float:
    class OracleNode:
        __slots__ = ("value", "left", "right")

        def __init__(self, value, left, right):
            self.value = value
            self.left = left
            self.right = right

    def measure(cls):
        total = 0
        for i in range(n):
            obj = cls(i, None, None)
            total += sys.getsizeof(obj)
            d = getattr(obj, "__dict__", None)
            if d is not None:
                total += sys.getsizeof(d)
        return total / n

    return measure(Node) / measure(OracleNode)
