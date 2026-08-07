import numpy as np


class MedusaHeads:
    """K=2 Medusa heads for predicting future tokens."""

    def __init__(self, hidden_dim, vocab_size, seed=42):
        raise NotImplementedError

    def forward(self, hidden_states):
        raise NotImplementedError

    def train_step(self, hidden_states, targets, lr=0.01):
        raise NotImplementedError
