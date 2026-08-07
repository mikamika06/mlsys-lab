import numpy as np


class LinearQLoRA:
    def __init__(self, in_features, out_features, r=4, lora_alpha=8, seed=42):
        rng = np.random.default_rng(seed)
        self.weight = rng.integers(-127, 127, size=(in_features, out_features), dtype=np.int8)
        self.scale = rng.random(size=(out_features,)).astype(np.float32) * 0.1
        self.lora_A = rng.normal(0, 0.01, size=(in_features, r)).astype(np.float32)
        self.lora_B = np.zeros((r, out_features), dtype=np.float32)
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = self.lora_alpha / self.r

    def forward(self, X):
        raise NotImplementedError

    def backward(self, X, grad_output, lr=0.01):
        raise NotImplementedError
