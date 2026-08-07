class PreemptionScheduler:
    def __init__(self, config, policy):
        raise NotImplementedError

    def step(self):
        raise NotImplementedError

    def run_trace(self, trace):
        raise NotImplementedError
