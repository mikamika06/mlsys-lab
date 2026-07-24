def my_islice(iterable, start: int, stop: int, step: int = 1):
    """Reimplementation of itertools.islice(iterable, start, stop, step):
    consume iterable one element at a time via iter/next, skip the first
    `start` elements, then yield every `step`-th element up to (not
    including) `stop`. No itertools, no eager materialization.
    """
    raise NotImplementedError('your code here')


def my_chain(*iterables):
    """Reimplementation of itertools.chain(*iterables): yield every
    element of each iterable in turn, consuming each one element at a
    time. No itertools, no eager materialization.
    """
    raise NotImplementedError('your code here')
