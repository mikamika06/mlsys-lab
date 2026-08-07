def ring_all_reduce(buffers: list[list[float]]) -> list[list[float]]:
    """
    Simulate ring all-reduce (reduce-scatter followed by all-gather)
    across N = len(buffers) ranks arranged in a ring, where rank i's
    neighbors are (i-1) % N and (i+1) % N.

    buffers: list of N 1-D arrays, all the same length L, with L
    divisible by N (each buffer splits evenly into N equal chunks).

    Return a list of N arrays: every rank's final buffer after the
    all-reduce, each equal to the elementwise sum of all N input
    buffers.
    """
    raise NotImplementedError('your code here')
