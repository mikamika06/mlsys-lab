class CompilationCache:
    def __init__(self):
        raise NotImplementedError

    def store(self, key, val):
        raise NotImplementedError

    def lookup(self, key):
        raise NotImplementedError

    def set_version(self, version):
        raise NotImplementedError

    def invalidate(self, key):
        raise NotImplementedError
