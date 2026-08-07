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
        dequant = self.weight.astype(np.float32) * self.scale
        base_out = X @ dequant
        lora_out = (X @ self.lora_A @ self.lora_B) * self.scaling
        return base_out + lora_out

    def backward(self, X, grad_output, lr=0.01):
        X_A = X @ self.lora_A
        grad_B = X_A.T @ grad_output * self.scaling
        grad_A = X.T @ (grad_output @ self.lora_B.T) * self.scaling
        self.lora_A -= lr * grad_A
        self.lora_B -= lr * grad_B

def train_20_steps(layer, X, target, lr=0.01):
    losses = []
    for _ in range(20):
        Y = layer.forward(X)
        loss = np.mean((Y - target)**2)
        losses.append(loss)
        grad_output = 2.0 * (Y - target) / Y.size
        layer.backward(X, grad_output, lr)
    return losses
