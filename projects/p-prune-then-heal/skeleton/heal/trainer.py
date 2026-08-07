import numpy as np


class HealerTrainer:
    """Fine-tunes a pruned model while maintaining parameter sparsity masks."""

    def __init__(self, model, mask_mgr, lr=0.02, max_steps=100):
        raise NotImplementedError

    def step(self, X_batch, y_batch):
        raise NotImplementedError

    def train(self, X, y, batch_size=32):
        raise NotImplementedError
