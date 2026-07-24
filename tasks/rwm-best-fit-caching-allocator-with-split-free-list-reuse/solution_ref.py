_BLOCK = 512


def _round_size(nbytes):
    if nbytes <= _BLOCK:
        return _BLOCK
    return _BLOCK * ((nbytes + _BLOCK - 1) // _BLOCK)


class CachingAllocator:
    """PyTorch-style caching allocator: round -> best-fit -> split -> grow-on-miss.

    Parameters
    ----------
    capacity : int
        Maximum total bytes that may ever be reserved from the device.
    """

    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.reserved = 0
        self._free = []          # list of {'id', 'size'}, insertion order
        self._live = {}          # id -> size
        self._next_id = 0

    def malloc(self, nbytes):
        """Request ``nbytes``. Returns a block id (int) on success, or None on OOM."""
        size = _round_size(int(nbytes))

        best_i = None
        for i, blk in enumerate(self._free):
            if blk["size"] >= size:
                if best_i is None or blk["size"] < self._free[best_i]["size"]:
                    best_i = i
        if best_i is not None:
            blk = self._free.pop(best_i)
            remainder = blk["size"] - size
            if remainder > 0:
                self._free.append({"id": self._new_id(), "size": remainder})
            bid = blk["id"]
            self._live[bid] = size
            return bid

        if self.reserved + size > self.capacity:
            return None

        self.reserved += size
        bid = self._new_id()
        self._live[bid] = size
        return bid

    def free(self, block_id):
        """Return a previously allocated block to the free-list pool."""
        size = self._live.pop(block_id)
        self._free.append({"id": self._new_id(), "size": size})

    def _new_id(self):
        bid = self._next_id
        self._next_id += 1
        return bid
