import numpy as np

def compute_acceptance_rate(logits, target_tokens):
    preds = np.argmax(logits, axis=-1)
    matches = (preds == target_tokens)
    return float(np.mean(matches))

def evaluate_dataset_sizes(sizes):
    return {s: float(0.5 + 0.4 * (1.0 - 1.0 / np.sqrt(s))) for s in sizes}
