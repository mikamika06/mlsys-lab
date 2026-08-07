import numpy as np

def compute_perplexity(logits, targets):
    exps = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = exps / np.sum(exps, axis=-1, keepdims=True)
    n = targets.shape[0]
    flat_targets = targets.flatten()
    flat_probs = probs.reshape(-1, probs.shape[-1])
    chosen = flat_probs[np.arange(n), flat_targets]
    nll = -np.log(np.maximum(chosen, 1e-12))
    return float(np.exp(np.mean(nll)))

def measure_perplexity(logits, targets, scale_factor):
    scaled_logits = logits / max(1.0, scale_factor * 0.2)
    return compute_perplexity(scaled_logits, targets)
