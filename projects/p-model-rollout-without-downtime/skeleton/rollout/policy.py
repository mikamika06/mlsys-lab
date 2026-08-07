class RolloutPolicy:
    def __init__(self, steps, error_threshold):
        raise NotImplementedError

    def get_weight(self, step):
        raise NotImplementedError

    def should_rollback(self, error_rate):
        raise NotImplementedError
