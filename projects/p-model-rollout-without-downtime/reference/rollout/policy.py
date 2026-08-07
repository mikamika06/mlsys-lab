class RolloutPolicy:
    def __init__(self, steps, error_threshold):
        self.steps = steps
        self.error_threshold = error_threshold
        self.current_step = 0
        self.rolled_back = False

    def get_weight(self, step):
        if self.rolled_back:
            return 0.0
        idx = min(step, len(self.steps) - 1)
        return self.steps[idx]

    def should_rollback(self, error_rate):
        if error_rate > self.error_threshold:
            self.rolled_back = True
            return True
        return False
