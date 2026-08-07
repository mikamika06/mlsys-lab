import numpy as np


class LoRALinear:
    def __init__(self, in_features, out_features, rank=4, alpha=1.0):
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        rng = np.random.default_rng(42)
        self.weight = rng.normal(0, 0.02, (out_features, in_features))
        self.lora_a = rng.normal(0, 0.02, (rank, in_features))
        self.lora_b = np.zeros((out_features, rank))

    def forward(self, x):
        base = x @ self.weight.T
        lora = (x @ self.lora_a.T) @ self.lora_b.T * (self.alpha / self.rank)
        return base + lora
