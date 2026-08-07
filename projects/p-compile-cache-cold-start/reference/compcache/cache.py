class CompilationCache:
    """In-memory cache for compiled artifacts."""
    def __init__(self):
        self.store_map = {}
        self.version = "v1"

    def store(self, key, val):
        self.store_map[key] = val

    def lookup(self, key):
        return self.store_map.get(key)

    def set_version(self, version):
        self.version = version

    def invalidate(self, key):
        if key in self.store_map:
            del self.store_map[key]
