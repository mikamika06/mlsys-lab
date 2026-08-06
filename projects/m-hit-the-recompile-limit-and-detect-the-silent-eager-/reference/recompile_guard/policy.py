from .detector import RecompileDetector


class GuardedFunction:
    def __init__(self, fn, recompile_limit=8):
        self.fn = fn
        self.limit = recompile_limit
        self.name = getattr(fn, "__name__", "unnamed")
        self.detector = RecompileDetector(limit=recompile_limit)
        self.detector.register(self.name)
        self._cache = set()

    def __call__(self, *args, **kwargs):
        sig = []
        for arg in args:
            if hasattr(arg, "shape"):
                sig.append(tuple(arg.shape))
            else:
                sig.append(type(arg))
        for k in sorted(kwargs.keys()):
            v = kwargs[k]
            if hasattr(v, "shape"):
                sig.append((k, tuple(v.shape)))
            else:
                sig.append((k, type(v)))
        key = tuple(sig)

        if key not in self._cache:
            if self.detector.get_stats(self.name)["compiles"] < self.limit:
                self._cache.add(key)
                self.detector.record_compile(self.name)

        self.detector.record_execution(self.name)
        return self.fn(*args, **kwargs)

    def status(self):
        return self.detector.get_stats(self.name)

    def reset(self):
        self._cache.clear()
        self.detector = RecompileDetector(limit=self.limit)
        self.detector.register(self.name)
