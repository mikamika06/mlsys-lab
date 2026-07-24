class CachingAllocator:
    """PyTorch-style caching allocator: round -> best-fit -> split -> grow-on-miss.

    Parameters
    ----------
    capacity : int
        Maximum total bytes that may ever be reserved from the device.

    Methods
    -------
    malloc(nbytes) -> int | None
        Request ``nbytes``. Returns a block id on success, or None on OOM.
    free(block_id) -> None
        Return a previously allocated block to the free-list pool.
    """

    def __init__(self, capacity):
        raise NotImplementedError('your code here')

    def malloc(self, nbytes):
        raise NotImplementedError('your code here')

    def free(self, block_id):
        raise NotImplementedError('your code here')
