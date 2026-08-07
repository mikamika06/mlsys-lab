import numpy as np


class EagleSampler:
    def __init__(self, topk=4):
        self.topk = topk

    def sample_tree(self, logits, temperature=1.0):
        scaled = logits / max(temperature, 1e-5)
        exp_logits = np.exp(scaled - np.max(scaled, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        top_indices = np.argsort(probs, axis=-1)[..., -self.topk:]
        return top_indices
