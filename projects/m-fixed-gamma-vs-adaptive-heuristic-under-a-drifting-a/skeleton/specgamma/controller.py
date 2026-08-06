class BaseController:
    def __init__(self, min_gamma: int = 1, max_gamma: int = 10, cost_ratio: float = 0.05):
        self.min_gamma = min_gamma
        self.max_gamma = max_gamma
        self.cost_ratio = cost_ratio

    def get_gamma(self) -> int:
        raise NotImplementedError

    def update(self, accepted_count: int, draft_gamma: int) -> None:
        raise NotImplementedError


class FixedGammaController(BaseController):
    def __init__(self, gamma: int, min_gamma: int = 1, max_gamma: int = 10, cost_ratio: float = 0.05):
        super().__init__(min_gamma=min_gamma, max_gamma=max_gamma, cost_ratio=cost_ratio)
        self.gamma = gamma

    def get_gamma(self) -> int:
        raise NotImplementedError

    def update(self, accepted_count: int, draft_gamma: int) -> None:
        raise NotImplementedError


class AdaptiveGammaController(BaseController):
    def __init__(self, ema_alpha: float = 0.2, min_gamma: int = 1, max_gamma: int = 10, cost_ratio: float = 0.05):
        super().__init__(min_gamma=min_gamma, max_gamma=max_gamma, cost_ratio=cost_ratio)
        self.ema_alpha = ema_alpha

    def get_gamma(self) -> int:
        raise NotImplementedError

    def update(self, accepted_count: int, draft_gamma: int) -> None:
        raise NotImplementedError
