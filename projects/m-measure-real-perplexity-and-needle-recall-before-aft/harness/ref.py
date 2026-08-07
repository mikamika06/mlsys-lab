import numpy as np

np.random.seed(42)

CONFIGS = [
    {"base_scale": 1.0, "seq_len": 2048, "target_len": 4096},
    {"base_scale": 2.0, "seq_len": 4096, "target_len": 8192},
    {"base_scale": 4.0, "seq_len": 4096, "target_len": 16384},
]

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

def measure_recall_at_k(retrieval_scores, needle_indices, k):
    top_k_indices = np.argsort(retrieval_scores, axis=-1)[:, -k:]
    hits = 0
    for i, needle in enumerate(needle_indices):
        if needle in top_k_indices[i]:
            hits += 1
    return float(hits / len(needle_indices))

def apply_scaling(positions, scale_factor):
    return positions / scale_factor
