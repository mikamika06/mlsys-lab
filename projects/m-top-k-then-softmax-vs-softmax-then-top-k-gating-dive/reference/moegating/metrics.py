import numpy as np


def compute_router_entropy(router_logits_per_layer: list[np.ndarray]) -> dict[str, np.ndarray]:
    per_layer_mean = []
    per_layer_per_token = []
    for logits in router_logits_per_layer:
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        probs = np.clip(probs, 1e-12, 1.0)
        entropy = -np.sum(probs * np.log(probs), axis=-1)
        per_layer_per_token.append(entropy)
        per_layer_mean.append(np.mean(entropy))
    return {
        "mean_entropy_per_layer": np.array(per_layer_mean),
        "entropy_per_token": np.array(per_layer_per_token),
    }
