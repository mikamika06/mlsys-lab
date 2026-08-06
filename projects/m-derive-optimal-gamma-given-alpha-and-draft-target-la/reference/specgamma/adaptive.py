class AdaptiveGamma:
    def __init__(self, initial_gamma: int = 4):
        self.gamma = initial_gamma
        self.history = []

    def update(self, accepted_count: int) -> int:
        self.history.append(accepted_count)
        if len(self.history) >= 3:
            recent = self.history[-3:]
            if all(a >= self.gamma for a in recent):
                self.gamma = min(8, self.gamma + 1)
            elif all(a < self.gamma - 1 for a in recent):
                self.gamma = max(1, self.gamma - 1)
        return self.gamma
