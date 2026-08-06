class FlopCounterMode:
    """Context manager to tally FLOPs."""
    _current = None

    def __init__(self):
        self.total = 0
        self._previous = None

    def __enter__(self):
        self._previous = FlopCounterMode._current
        FlopCounterMode._current = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        FlopCounterMode._current = self._previous

    @classmethod
    def record(cls, flops):
        if cls._current is not None:
            cls._current.total += flops
