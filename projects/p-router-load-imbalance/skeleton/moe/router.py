import numpy as np


class MoERouter:

    def __init__(self, num_experts, in_dim):
        raise NotImplementedError

    def route(self, x, top_k=2):
        raise NotImplementedError

    def compute_aux_loss(self, router_probs, selected_experts):
        raise NotImplementedError

    def update_weights(self, grads, lr=0.01):
        raise NotImplementedError


def simulate_step_time(expert_counts, capacity_per_expert=100, base_cost=1.0):
    raise NotImplementedError
