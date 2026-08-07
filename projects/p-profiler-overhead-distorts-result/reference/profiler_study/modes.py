class ProfilerEnvironment:
    def __init__(self, base_cost: float = 120.0):
        self.base_cost = base_cost
        self.overheads = {
            "clean": 0.0,
            "sampling": 5.0,
            "instrumentation": 60.0
        }

    def measure(self, mode: str) -> float:
        if mode not in self.overheads:
            raise ValueError("unknown mode")
        return self.base_cost + self.overheads[mode]
