import numpy as np

class DraftModel:
    def __init__(self, vocab_size: int, hidden_size: int):
        self.vocab_size = vocab_size
        np.random.seed(42)
        self.W1 = np.random.randn(vocab_size, hidden_size) * 0.1
        self.W2 = np.random.randn(hidden_size, vocab_size) * 0.1

    def forward(self, token: int):
        return self.W1[token] @ self.W2

    def get_probs(self, token: int):
        logits = self.forward(token)
        logits -= np.max(logits)
        exp = np.exp(logits)
        return exp / np.sum(exp)
