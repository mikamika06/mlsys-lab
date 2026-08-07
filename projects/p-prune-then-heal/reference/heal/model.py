import numpy as np


class SimpleMLP:
    """A two-layer neural network for multi-class classification."""

    def __init__(self, input_dim=20, hidden_dim=40, output_dim=5, seed=42):
        rng = np.random.RandomState(seed)
        self.W1 = rng.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.randn(hidden_dim, output_dim) * 0.1
        self.b2 = np.zeros(output_dim)

    @property
    def weights(self):
        return [self.W1, self.W2]

    def forward_backward(self, X, y):
        z1 = X @ self.W1 + self.b1
        a1 = np.maximum(0, z1)
        z2 = a1 @ self.W2 + self.b2

        exp_z2 = np.exp(z2 - np.max(z2, axis=1, keepdims=True))
        probs = exp_z2 / np.sum(exp_z2, axis=1, keepdims=True)

        N = X.shape[0]
        loss = -np.mean(np.log(probs[np.arange(N), y] + 1e-12))

        dz2 = probs.copy()
        dz2[np.arange(N), y] -= 1.0
        dz2 /= N

        dW2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0)

        da1 = dz2 @ self.W2.T
        dz1 = da1 * (z1 > 0)
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0)

        return float(loss), [dW1, dW2], [db1, db2]

    def apply_gradients(self, weight_grads, bias_grads, lr):
        self.W1 -= lr * weight_grads[0]
        self.W2 -= lr * weight_grads[1]
        self.b1 -= lr * bias_grads[0]
        self.b2 -= lr * bias_grads[1]

    def evaluate(self, X, y):
        z1 = X @ self.W1 + self.b1
        a1 = np.maximum(0, z1)
        z2 = a1 @ self.W2 + self.b2
        preds = np.argmax(z2, axis=1)
        acc = np.mean(preds == y)
        exp_z2 = np.exp(z2 - np.max(z2, axis=1, keepdims=True))
        probs = exp_z2 / np.sum(exp_z2, axis=1, keepdims=True)
        loss = -np.mean(np.log(probs[np.arange(y.shape[0]), y] + 1e-12))
        return float(acc), float(loss)
