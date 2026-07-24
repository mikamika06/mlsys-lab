class _CountdownMachine:
    def __init__(self, n):
        self.state = n

    def __next__(self):
        self.state -= 1
        if self.state < 0:
            raise StopIteration
        return self.state


def countdown(n):
    return _CountdownMachine(n)
