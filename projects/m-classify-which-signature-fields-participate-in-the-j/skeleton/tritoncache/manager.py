"""Kernel JIT Cache Manager."""


class JITCacheManager:
    """Manages kernel compilation cache and dispatch tracking."""

    def __init__(self, fn_name, sig_spec):
        raise NotImplementedError

    def get_or_compile(self, args_kw):
        """Looks up or compiles kernel based on argument signature."""
        raise NotImplementedError

    def stats(self):
        """Returns hit and miss counts."""
        raise NotImplementedError
