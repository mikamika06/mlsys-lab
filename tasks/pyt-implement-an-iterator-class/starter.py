class CountdownIterator:
    def __init__(self, start: int):
        self.current = start

    def __iter__(self):
        # TODO: this returns a new iterable instead of following the iterator protocol
        return range(self.current, 0, -1)

    def __next__(self):
        # TODO: missing state update and StopIteration handling
        return self.current
