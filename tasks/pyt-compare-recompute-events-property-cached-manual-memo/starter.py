import functools


def expensive(n: int) -> int:
    """Sum of squares 0^2 + 1^2 + ... + (n-1)^2, via an explicit loop."""
    raise NotImplementedError('your code here')


class PropertyDemo:
    """.value is a plain @property that calls expensive(self.n) every access."""

    def __init__(self, n: int):
        raise NotImplementedError('your code here')


class CachedPropertyDemo:
    """.value is a functools.cached_property wrapping expensive(self.n)."""

    def __init__(self, n: int):
        raise NotImplementedError('your code here')


class ManualMemoDemo:
    """.value is a property that manually memoizes expensive(self.n)."""

    def __init__(self, n: int):
        raise NotImplementedError('your code here')
