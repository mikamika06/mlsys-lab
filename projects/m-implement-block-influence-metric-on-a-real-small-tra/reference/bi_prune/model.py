import numpy as np


class SmallTransformer:
    def __init__(self, num_layers, hidden_dim, seed=42):
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        rng = np.random.default_rng(seed)
        self.weights = []
        for _ in range(num_layers):
            w1 = rng.normal(0, 0.1, (hidden_dim, hidden_dim))
            b1 = np.zeros(hidden_dim)
            self.weights.append((w1, b1))
        self.head = rng.normal(0, 0.1, (hidden_dim, hidden_dim))

    def forward(self, x, return_activations=False):
        activations = [x]
        curr = x
        for w1, b1 in self.weights:
            curr = curr + np.tanh(np.dot(curr, w1) + b1)
            activations.append(curr)
        out = np.dot(curr, self.head)
        if return_activations:
            return out, activations
        return out
