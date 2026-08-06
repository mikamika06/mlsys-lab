import numpy as np
import math


CONFIGS = [
    {
        "d_model": 256,
        "d_ffn_coarse": 1024,
        "num_coarse": 8,
        "k_coarse": 2,
        "num_shared": 2,
        "k_fine": 6,
        "split_factor": 4,
        "num_routed_fine": 32
    },
    {
        "d_model": 512,
        "d_ffn_coarse": 2048,
        "num_coarse": 16,
        "k_coarse": 2,
        "num_shared": 4,
        "k_fine": 12,
        "split_factor": 8,
        "num_routed_fine": 128
    }
]


def derive_fine_grained_split(d_model, d_ffn_coarse, num_coarse, k_coarse, num_shared, k_fine, split_factor):
    d_ffn_fine = int(d_ffn_coarse // split_factor)
    flops_coarse_per_token = k_coarse * 3 * 2 * d_model * d_ffn_coarse
    flops_fine_routed_per_token = k_fine * 3 * 2 * d_model * d_ffn_fine
    flops_shared_per_token = num_shared * 3 * 2 * d_model * d_ffn_fine
    flops_fine_total_per_token = flops_fine_routed_per_token + flops_shared_per_token

    return {
        "d_ffn_fine": d_ffn_fine,
        "coarse_flops_per_token": flops_coarse_per_token,
        "fine_flops_per_token": flops_fine_total_per_token,
        "is_equivalent": flops_coarse_per_token == flops_fine_total_per_token
    }


def count_reachable_combinations(num_routed, top_k, num_shared=0):
    if top_k > num_routed:
        return 0
    return math.comb(num_routed, top_k)


class ReferenceMoEBlock:
    def __init__(self, d_model, d_ffn_fine, num_shared, num_routed, top_k, seed=42):
        np.random.seed(seed)
        self.d_model = d_model
        self.d_ffn_fine = d_ffn_fine
        self.num_shared = num_shared
        self.num_routed = num_routed
        self.top_k = top_k

        self.w_gate = np.random.randn(d_model, num_routed) * 0.02

        self.shared_w1 = [np.random.randn(d_model, d_ffn_fine) * 0.02 for _ in range(num_shared)]
        self.shared_w2 = [np.random.randn(d_ffn_fine, d_model) * 0.02 for _ in range(num_shared)]
        self.shared_w3 = [np.random.randn(d_model, d_ffn_fine) * 0.02 for _ in range(num_shared)]

        self.routed_w1 = [np.random.randn(d_model, d_ffn_fine) * 0.02 for _ in range(num_routed)]
        self.routed_w2 = [np.random.randn(d_ffn_fine, d_model) * 0.02 for _ in range(num_routed)]
        self.routed_w3 = [np.random.randn(d_model, d_ffn_fine) * 0.02 for _ in range(num_routed)]

    def _swiglu(self, x, w1, w2, w3):
        h1 = np.matmul(x, w1)
        h3 = np.matmul(x, w3)
        silu = h1 * (1.0 / (1.0 + np.exp(-np.clip(h1, -50, 50))))
        return np.matmul(silu * h3, w2)

    def route(self, x):
        logits = np.matmul(x, self.w_gate)
        top_k_indices = np.argsort(logits, axis=-1)[:, -self.top_k:]
        top_k_logits = np.take_along_axis(logits, top_k_indices, axis=-1)
        exp_logits = np.exp(top_k_logits - np.max(top_k_logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        return top_k_indices, probs

    def forward(self, x):
        batch_size = x.shape[0]
        out = np.zeros_like(x)

        for i in range(self.num_shared):
            out += self._swiglu(x, self.shared_w1[i], self.shared_w2[i], self.shared_w3[i])

        indices, probs = self.route(x)
        for b in range(batch_size):
            for k in range(self.top_k):
                exp_idx = indices[b, k]
                weight = probs[b, k]
                expert_out = self._swiglu(
                    x[b:b+1],
                    self.routed_w1[exp_idx],
                    self.routed_w2[exp_idx],
                    self.routed_w3[exp_idx]
                )
                out[b] += weight * expert_out[0]

        return out
