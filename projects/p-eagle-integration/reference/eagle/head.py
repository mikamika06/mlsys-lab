import numpy as np


class DraftHead:
    def __init__(self, hidden_dim, vocab_size, seed=42):
        rng = np.random.default_rng(seed)
        self.w = rng.normal(0, 0.02, (hidden_dim, vocab_size))
        self.b = np.zeros(vocab_size)

    def forward(self, hidden_states):
        return np.dot(hidden_states, self.w) + self.b
