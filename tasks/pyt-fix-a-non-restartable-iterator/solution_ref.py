class _SquaresIterator:
    def __init__(self, n):
        self.n = n
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.i >= self.n:
            raise StopIteration
        val = self.i * self.i
        self.i += 1
        return val


class Squares:
    """Iterable over 0**2, 1**2, ..., (n-1)**2. Restartable: __iter__ returns a fresh iterator."""

    def __init__(self, n):
        self.n = n

    def __iter__(self):
        return _SquaresIterator(self.n)
