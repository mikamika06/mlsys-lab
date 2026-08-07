from routing.policy import CacheAwarePolicy


class Router:
    def __init__(self, num_replicas):
        self.num_replicas = num_replicas
        self.policy = CacheAwarePolicy(num_replicas)
        self.replica_states = [set() for _ in range(num_replicas)]

    def round_robin_route(self, req_idx):
        return req_idx % self.num_replicas

    def step(self, prompt, use_round_robin=False, replica_states=None):
        states = replica_states if replica_states is not None else self.replica_states
        if use_round_robin:
            return 0
        return self.policy.route(prompt, states)
