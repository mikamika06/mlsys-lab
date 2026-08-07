def evaluate_policy(requests, policy_fn, cost_model, gamma):
    raise NotImplementedError

class AdaptivePolicy:
    def __init__(self, cost_model, gamma, default_p=0.5):
        raise NotImplementedError

    def update(self, domain, drafted, accepted):
        raise NotImplementedError

    def decide(self, domain, b):
        raise NotImplementedError
