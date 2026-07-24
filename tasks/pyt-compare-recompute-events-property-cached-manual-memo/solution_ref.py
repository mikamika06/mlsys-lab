import functools


def expensive(n: int) -> int:
    """Sum of squares 0^2 + 1^2 + ... + (n-1)^2, via an explicit loop."""
    total = 0
    for i in range(n):
        total += i * i
    return total


class PropertyDemo:
    """.value is a plain @property that calls expensive(self.n) every access."""

    def __init__(self, n: int):
        self.n = n

    @property
    def value(self):
        return expensive(self.n)


class CachedPropertyDemo:
    """.value is a functools.cached_property wrapping expensive(self.n)."""

    def __init__(self, n: int):
        self.n = n

    @functools.cached_property
    def value(self):
        return expensive(self.n)


class ManualMemoDemo:
    """.value is a property that manually memoizes expensive(self.n)."""

    def __init__(self, n: int):
        self.n = n
        self._cached = None

    @property
    def value(self):
        if self._cached is None:
            self._cached = expensive(self.n)
        return self._cached
