import sys


class _OracleNode:
    __slots__ = ("value", "left", "right")

    def __init__(self, value, left, right):
        self.value = value
        self.left = left
        self.right = right


def _measure(cls, n):
    total = 0
    for i in range(n):
        obj = cls(i, None, None)
        total += sys.getsizeof(obj)
        d = getattr(obj, "__dict__", None)
        if d is not None:
            total += sys.getsizeof(d)
    return total / n


def grade(sol, fx) -> dict:
    n = 100000
    try:
        oracle_bytes = _measure(_OracleNode, n)
        candidate_bytes = _measure(sol.Node, n)
        ratio = candidate_bytes / oracle_bytes

        reported = sol.node_size_ratio(n)
        if abs(float(reported) - ratio) > 1e-12:
            return {"size_ratio": float("inf")}

        return {"size_ratio": float(ratio)}
    except Exception:
        return {"size_ratio": float("inf")}
