def make_counter(start=0):
    """
    Return a callable that increments an internal counter each time it is called.
    The first call returns start + 1, subsequent calls return successive integers.
    """
    counter = start

    def inc():
        nonlocal counter
        counter += 1
        return counter

    return inc
