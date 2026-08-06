class AdaptiveGamma:
    def __init__(self, initial_gamma: int = 4):
        raise NotImplementedError

    def update(self, accepted_count: int) -> int:
        raise NotImplementedError
