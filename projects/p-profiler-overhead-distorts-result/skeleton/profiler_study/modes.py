class ProfilerEnvironment:
    def __init__(self, base_cost: float = 120.0):
        raise NotImplementedError

    def measure(self, mode: str) -> float:
        raise NotImplementedError
