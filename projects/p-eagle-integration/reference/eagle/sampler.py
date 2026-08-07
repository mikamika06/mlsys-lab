import numpy as np


class DraftSampler:
    def __init__(self, temperature=1.0):
        self.temperature = max(temperature, 1e-5)

    def sample(self, logits):
        if isinstance(logits, list):
            logits = np.array(logits, dtype=np.float32)
        scaled = logits / self.temperature
        exps = np.exp(scaled - np.max(scaled, axis=-1, keepdims=True))
        probs = exps / np.sum(exps, axis=-1, keepdims=True)
        return int(np.argmax(probs))
