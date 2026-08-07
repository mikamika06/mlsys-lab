import numpy as np


class MoERouter:

    def __init__(self, num_experts, in_dim):
        self.num_experts = num_experts
        self.in_dim = in_dim
        np.random.seed(42)
        self.W = np.random.randn(in_dim, num_experts) * 0.1

    def route(self, x, top_k=2):
        logits = np.dot(x, self.W)
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        top_k_indices = np.argsort(probs, axis=-1)[:, -top_k:][:, ::-1]
        top_k_weights = np.take_along_axis(probs, top_k_indices, axis=-1)
        top_k_weights = top_k_weights / np.sum(
            top_k_weights, axis=-1, keepdims=True
        )

        return probs, top_k_indices, top_k_weights

    def compute_aux_loss(self, router_probs, selected_experts):
        N = router_probs.shape[0]
        E = self.num_experts

        P = np.mean(router_probs, axis=0)

        counts = np.zeros(E)
        for idx in selected_experts.flat:
            counts[idx] += 1
        f = counts / (N * selected_experts.shape[1])

        aux_loss = E * np.sum(f * P)
        grad_W_aux = np.zeros_like(self.W)

        for i in range(E):
            d_P_i = np.mean(
                router_probs[:, i : i + 1]
                * (
                    np.eye(E)[i]
                    - router_probs
                ),
                axis=0,
            )
        return float(aux_loss), grad_W_aux

    def update_weights(self, grads, lr=0.01):
        self.W -= lr * grads


def simulate_step_time(expert_counts, capacity_per_expert=100, base_cost=1.0):
    overflow = np.maximum(0, expert_counts - capacity_per_expert)
    max_tokens = np.max(expert_counts)
    step_time = base_cost * max_tokens + 0.05 * np.sum(overflow)
    return float(step_time)
