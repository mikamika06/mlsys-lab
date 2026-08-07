class DynamicBatchController:
    def __init__(self, profile, slo):
        raise NotImplementedError
    def step(self, incoming, current_load):
        raise NotImplementedError
