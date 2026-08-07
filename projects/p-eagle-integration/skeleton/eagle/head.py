import numpy as np


class DraftHead:
    def __init__(self, hidden_dim, vocab_size):
        raise NotImplementedError

    def forward(self, hidden_states):
        raise NotImplementedError
