import numpy as np

def temperature_scale(logits, T):
    logits = np.asarray(logits, dtype=np.float64)
    scaled = logits / T
    max_scaled = np.max(scaled, axis=-1, keepdims=True)
    exp_scaled = np.exp(scaled - max_scaled)
    probs = exp_scaled / np.sum(exp_scaled, axis=-1, keepdims=True)
    return probs
