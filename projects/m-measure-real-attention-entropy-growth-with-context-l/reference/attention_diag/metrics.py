import numpy as np


def compute_attention_entropy(attention_weights):
    eps = 1e-12
    p = np.clip(attention_weights, eps, 1.0)
    p = p / np.sum(p, axis=-1, keepdims=True)
    entropy = -np.sum(p * np.log2(p), axis=-1)
    return float(np.mean(entropy))
