class EagerFallbackDetector:
    def __init__(self, limit=8):
        raise NotImplementedError

    def step(self, is_compiled, guard_id=None):
        raise NotImplementedError
