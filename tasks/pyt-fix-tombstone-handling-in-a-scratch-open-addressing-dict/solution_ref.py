_EMPTY = object()
_DELETED = object()


class ScratchDict:
    """Fixed-capacity open-addressing dict with linear probing and correct
    tombstone handling: probing skips DELETED slots but stops at a real
    EMPTY slot, and insertion is free to reuse the first tombstone it sees.
    """

    def __init__(self, capacity: int = 16):
        self.capacity = capacity
        self._slots = [_EMPTY] * capacity
        self._keys = [None] * capacity
        self._vals = [None] * capacity

    def _probe(self, key, for_insert: bool):
        idx = hash(key) % self.capacity
        first_tombstone = None
        for _ in range(self.capacity):
            slot = self._slots[idx]
            if slot is _EMPTY:
                if for_insert and first_tombstone is not None:
                    return first_tombstone, False
                return idx, False
            if slot is _DELETED:
                if first_tombstone is None:
                    first_tombstone = idx
            elif self._keys[idx] == key:
                return idx, True
            idx = (idx + 1) % self.capacity
        raise RuntimeError("ScratchDict is full")

    def set(self, key: int, value) -> None:
        idx, _found = self._probe(key, for_insert=True)
        self._slots[idx] = True
        self._keys[idx] = key
        self._vals[idx] = value

    def get(self, key: int):
        idx, found = self._probe(key, for_insert=False)
        if not found:
            raise KeyError(key)
        return self._vals[idx]

    def delete(self, key: int) -> None:
        idx, found = self._probe(key, for_insert=False)
        if not found:
            raise KeyError(key)
        self._slots[idx] = _DELETED
        self._keys[idx] = None
        self._vals[idx] = None

    def __contains__(self, key: int) -> bool:
        try:
            self.get(key)
            return True
        except KeyError:
            return False
