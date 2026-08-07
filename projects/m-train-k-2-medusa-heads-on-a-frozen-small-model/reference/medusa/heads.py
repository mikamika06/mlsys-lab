import numpy as np


def _softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)


class MedusaHeads:
    """K=2 Medusa heads for predicting future tokens."""

    def __init__(self, hidden_dim, vocab_size, seed=42):
        rng = np.random.RandomState(seed)
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.W1 = [
            rng.randn(hidden_dim, hidden_dim) * 0.02,
            rng.randn(hidden_dim, hidden_dim) * 0.02,
        ]
        self.b1 = [
            np.zeros(hidden_dim),
            np.zeros(hidden_dim),
        ]
        self.W2 = [
            rng.randn(hidden_dim, vocab_size) * 0.02,
            rng.randn(hidden_dim, vocab_size) * 0.02,
        ]
        self.b2 = [
            np.zeros(vocab_size),
            np.zeros(vocab_size),
        ]

    def forward(self, hidden_states):
        logits = []
        for k in range(2):
            h = np.maximum(0, np.matmul(hidden_states, self.W1[k]) + self.b1[k])
            h = h + hidden_states
            l = np.matmul(h, self.W2[k]) + self.b2[k]
            logits.append(l)
        return logits

    def train_step(self, hidden_states, targets, lr=0.01):
        N, T, D = hidden_states.shape
        V = self.vocab_size
        total_loss = 0.0

        for k in range(2):
            shift = k + 1
            if T <= shift:
                continue
            h_in = hidden_states[:, : T - shift, :]
            tgt = targets[:, shift:T]

            flat_h_in = h_in.reshape(-1, D)
            flat_tgt = tgt.reshape(-1)
            M = flat_h_in.shape[0]

            h1 = np.maximum(0, np.matmul(flat_h_in, self.W1[k]) + self.b1[k])
            res = h1 + flat_h_in
            logits = np.matmul(res, self.W2[k]) + self.b2[k]
            probs = _softmax(logits)

            loss = -np.log(probs[np.arange(M), flat_tgt] + 1e-12).mean()
            total_loss += loss

            dlogits = probs.copy()
            dlogits[np.arange(M), flat_tgt] -= 1.0
            dlogits /= M

            dW2 = np.matmul(res.T, dlogits)
            db2 = np.sum(dlogits, axis=0)

            dres = np.matmul(dlogits, self.W2[k].T)
            dh1 = dres * (h1 > 0)

            dW1 = np.matmul(flat_h_in.T, dh1)
            db1 = np.sum(dh1, axis=0)

            self.W2[k] -= lr * dW2
            self.b2[k] -= lr * db2
            self.W1[k] -= lr * dW1
            self.b1[k] -= lr * db1

        return total_loss
