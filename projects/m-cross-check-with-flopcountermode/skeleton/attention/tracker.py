class FlopCounterMode:
    """Context manager to tally FLOPs."""
    _current = None

    def __init__(self):
        raise NotImplementedError

    def __enter__(self):
        raise NotImplementedError

    def __exit__(self, exc_type, exc_val, exc_tb):
        raise NotImplementedError

    @classmethod
    def record(cls, flops):
        raise NotImplementedError
