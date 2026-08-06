class BaseController:
    def __init__(self, min_gamma: int = 1, max_gamma: int = 10, cost_ratio: float = 0.05):
        self.min_gamma = min_gamma
        self.max_gamma = max_gamma
        self.cost_ratio = cost_ratio

    def get_gamma(self) -> int:
        raise NotImplementedError

    def update(self, accepted_count: int, draft_gamma: int) -> None:
        pass


class FixedGammaController(BaseController):
    def __init__(self, gamma: int, min_gamma: int = 1, max_gamma: int = 10, cost_ratio: float = 0.05):
        super().__init__(min_gamma=min_gamma, max_gamma=max_gamma, cost_ratio=cost_ratio)
        self.gamma = max(self.min_gamma, min(gamma, self.max_gamma))

    def get_gamma(self) -> int:
        return self.gamma

    def update(self, accepted_count: int, draft_gamma: int) -> None:
        pass


class AdaptiveGammaController(BaseController):
    def __init__(self, ema_alpha: float = 0.2, min_gamma: int = 1, max_gamma: int = 10, cost_ratio: float = 0.05):
        super().__init__(min_gamma=min_gamma, max_gamma=max_gamma, cost_ratio=cost_ratio)
        self.ema_alpha = ema_alpha
        self.alpha_est = 0.5

    def get_gamma(self) -> int:
        best_g = self.min_gamma
        best_tput = -1.0
        for g in range(self.min_gamma, self.max_gamma + 1):
            if abs(self.alpha_est - 1.0) < 1e-9:
                exp_tokens = float(g + 1)
            else:
                exp_tokens = self.alpha_est * (1.0 - self.alpha_est**g) / (1.0 - self.alpha_est) + 1.0
            cost = 1.0 + g * self.cost_ratio
            tput = exp_tokens / cost
            if tput > best_tput + 1e-12:
                best_tput = tput
                best_g = g
        return best_g

    def update(self, accepted_count: int, draft_gamma: int) -> None:
        if draft_gamma <= 0:
            return
        observed = float(accepted_count) / float(draft_gamma)
        self.alpha_est = (1.0 - self.ema_alpha) * self.alpha_est + self.ema_alpha * observed
