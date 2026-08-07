import numpy as np


class DraftHead:
    def __init__(self, hidden_dim, vocab_size):
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.weight = np.zeros((hidden_dim, vocab_size), dtype=np.float32)

    def forward(self, hidden_states):
        if isinstance(hidden_states, list):
            hidden_states = np.array(hidden_states, dtype=np.float32)
        return np.dot(hidden_states, self.weight)
