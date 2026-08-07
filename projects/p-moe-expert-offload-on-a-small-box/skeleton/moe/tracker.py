class ActivationTracker:
    def __init__(self, num_experts):
        raise NotImplementedError

    def update(self, selected_experts):
        raise NotImplementedError

    def get_distribution(self):
        raise NotImplementedError
