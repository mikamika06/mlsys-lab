"""Kernel JIT Cache Manager."""

from tritoncache.cache_key import build_cache_key


class JITCacheManager:
    """Manages kernel compilation cache and dispatch tracking."""

    def __init__(self, fn_name, sig_spec):
        self.fn_name = fn_name
        self.sig_spec = sig_spec
        self.cache = {}
        self.hits = 0
        self.misses = 0

    def get_or_compile(self, args_kw):
        """Looks up or compiles kernel based on argument signature."""
        key = build_cache_key(self.fn_name, self.sig_spec, args_kw)
        if key in self.cache:
            self.hits += 1
            return self.cache[key], False
        else:
            self.misses += 1
            compiled_kernel = f"compiled_{self.fn_name}_{len(self.cache)}"
            self.cache[key] = compiled_kernel
            return compiled_kernel, True

    def stats(self):
        """Returns hit and miss counts."""
        return {"hits": self.hits, "misses": self.misses, "cache_size": len(self.cache)}
