def my_islice(iterable, start: int, stop: int, step: int = 1):
    """Reimplementation of itertools.islice, one element at a time."""
    it = iter(iterable)
    i = 0
    while i < start:
        next(it)
        i += 1
    while i < stop:
        try:
            val = next(it)
        except StopIteration:
            return
        yield val
        i += 1
        for _ in range(step - 1):
            try:
                next(it)
            except StopIteration:
                return
            i += 1


def my_chain(*iterables):
    """Reimplementation of itertools.chain, one element at a time."""
    for iterable in iterables:
        it = iter(iterable)
        for val in it:
            yield val
