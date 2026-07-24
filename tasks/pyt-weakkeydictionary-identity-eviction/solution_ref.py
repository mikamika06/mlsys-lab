import weakref


class IdentityCache:
    """Identity-keyed cache; an entry auto-evicts when its key object dies."""

    def __init__(self):
        self._store = weakref.WeakKeyDictionary()

    def put(self, key, value):
        self._store[key] = value

    def get(self, key, default=None):
        return self._store.get(key, default)

    def __len__(self):
        return len(self._store)
