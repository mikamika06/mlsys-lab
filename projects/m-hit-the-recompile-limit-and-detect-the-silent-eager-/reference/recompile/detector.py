class EagerFallbackDetector:
    def __init__(self, limit=8):
        self.limit = limit
        self.recompiles = 0
        self.fallback = False

    def step(self, is_compiled, guard_id=None):
        if not is_compiled:
            self.fallback = True
        else:
            self.recompiles += 1
            if self.recompiles > self.limit:
                self.fallback = True
        return self.fallback
