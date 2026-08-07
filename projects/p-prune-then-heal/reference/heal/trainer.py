import numpy as np


class HealerTrainer:
    """Fine-tunes a pruned model while maintaining parameter sparsity masks."""

    def __init__(self, model, mask_mgr, lr=0.02, max_steps=100):
        self.model = model
        self.mask_mgr = mask_mgr
        self.lr = lr
        self.max_steps = max_steps
        self.step_count = 0
        self.history = []

    def step(self, X_batch, y_batch):
        if self.step_count >= self.max_steps:
            raise RuntimeError("Step budget exceeded")
        loss, weight_grads, bias_grads = self.model.forward_backward(X_batch, y_batch)
        self.mask_mgr.mask_gradients(weight_grads)
        self.model.apply_gradients(weight_grads, bias_grads, self.lr)
        self.mask_mgr.apply_mask()
        self.step_count += 1
        self.history.append(float(loss))
        return float(loss)

    def train(self, X, y, batch_size=32):
        n_samples = X.shape[0]
        rng = np.random.RandomState(42)
        while self.step_count < self.max_steps:
            indices = rng.choice(n_samples, size=min(batch_size, n_samples), replace=False)
            self.step(X[indices], y[indices])
        return self.history
