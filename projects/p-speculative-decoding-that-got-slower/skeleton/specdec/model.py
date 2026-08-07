class SpeculativeModel:
    def __init__(self, target_step_cost=10.0, draft_step_cost=1.0, overhead_per_draft=0.1):
        raise NotImplementedError

    def expected_accepted_tokens(self, gamma, tau):
        raise NotImplementedError

    def expected_step_cost(self, gamma, batch_size=1):
        raise NotImplementedError

    def expected_speedup(self, gamma, tau, batch_size=1):
        raise NotImplementedError

    def optimal_gamma(self, tau, max_gamma=8, batch_size=1):
        raise NotImplementedError

    def batched_crossover_point(self, gamma, tau, max_batch=64):
        raise NotImplementedError
