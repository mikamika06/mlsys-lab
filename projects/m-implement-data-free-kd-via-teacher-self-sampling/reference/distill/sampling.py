import numpy as np


def teacher_self_sample(teacher_weights, num_samples, seed=42):
    rng = np.random.default_rng(seed)
    dim = teacher_weights.shape[0]
    latents = rng.normal(0.0, 1.0, size=(num_samples, dim))
    logits = latents @ teacher_weights
    probs = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs /= np.sum(probs, axis=-1, keepdims=True)
    return probs
