_EMPTY = object()


class ScratchDict:
    """Fixed-capacity open-addressing dict with linear probing.

    BUGGY: delete() blanks a slot straight back to EMPTY. That breaks the
    probe chain for every later key that collided through this slot -- a
    lookup for such a key stops early at the "empty" hole and reports it
    missing even though it is still stored further down the chain. Fix it
    with a proper tombstone (see task.md).
    """

    def __init__(self, capacity: int = 16):
        self.capacity = capacity
        self._slots = [_EMPTY] * capacity
        self._keys = [None] * capacity
        self._vals = [None] * capacity

    def _find_for_lookup(self, key):
        idx = hash(key) % self.capacity
        for _ in range(self.capacity):
            slot = self._slots[idx]
            if slot is _EMPTY:
                return idx, False
            if self._keys[idx] == key:
                return idx, True
            idx = (idx + 1) % self.capacity
        return None, False

    def _find_for_insert(self, key):
        idx = hash(key) % self.capacity
        for _ in range(self.capacity):
            slot = self._slots[idx]
            if slot is _EMPTY or self._keys[idx] == key:
                return idx
            idx = (idx + 1) % self.capacity
        raise RuntimeError("ScratchDict is full")

    def set(self, key: int, value) -> None:
        idx = self._find_for_insert(key)
        self._slots[idx] = True
        self._keys[idx] = key
        self._vals[idx] = value

    def get(self, key: int):
        idx, found = self._find_for_lookup(key)
        if not found:
            raise KeyError(key)
        return self._vals[idx]

    def delete(self, key: int) -> None:
        idx, found = self._find_for_lookup(key)
        if not found:
            raise KeyError(key)
        # BUG: this blanks the slot straight to EMPTY instead of a tombstone,
        # which breaks the probe chain for any later-colliding key.
        self._slots[idx] = _EMPTY
        self._keys[idx] = None
        self._vals[idx] = None

    def __contains__(self, key: int) -> bool:
        try:
            self.get(key)
            return True
        except KeyError:
            return False
