class SpeculativeModel:
    def __init__(self, target_step_cost=10.0, draft_step_cost=1.0, overhead_per_draft=0.1):
        self.target_step_cost = float(target_step_cost)
        self.draft_step_cost = float(draft_step_cost)
        self.overhead_per_draft = float(overhead_per_draft)

    def expected_accepted_tokens(self, gamma, tau):
        tau = float(tau)
        gamma = int(gamma)
        if tau >= 1.0:
            return float(gamma + 1)
        if tau <= 0.0:
            return 1.0
        return float((1.0 - (tau ** (gamma + 1))) / (1.0 - tau))

    def expected_step_cost(self, gamma, batch_size=1):
        batch_factor = 1.0 + 0.05 * (batch_size - 1)
        draft_cost = gamma * (self.draft_step_cost * batch_factor + self.overhead_per_draft)
        target_verify_cost = self.target_step_cost * (1.0 + 0.02 * gamma) * batch_factor
        return draft_cost + target_verify_cost

    def expected_speedup(self, gamma, tau, batch_size=1):
        if gamma <= 0:
            return 1.0
        exp_tokens = self.expected_accepted_tokens(gamma, tau)
        spec_cost = self.expected_step_cost(gamma, batch_size)
        base_batch_factor = 1.0 + 0.05 * (batch_size - 1)
        baseline_cost_per_token = self.target_step_cost * base_batch_factor
        cost_per_accepted_token = spec_cost / exp_tokens
        return baseline_cost_per_token / cost_per_accepted_token

    def optimal_gamma(self, tau, max_gamma=8, batch_size=1):
        best_gamma = 0
        best_speedup = 1.0
        for g in range(1, max_gamma + 1):
            s = self.expected_speedup(g, tau, batch_size)
            if s > best_speedup:
                best_speedup = s
                best_gamma = g
        return best_gamma

    def batched_crossover_point(self, gamma, tau, max_batch=64):
        for b in range(1, max_batch + 1):
            s = self.expected_speedup(gamma, tau, b)
            if s < 1.0:
                return b
        return max_batch + 1
