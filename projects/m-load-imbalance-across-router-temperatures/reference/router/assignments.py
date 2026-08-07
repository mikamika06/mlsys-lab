import numpy as np


def analyze_router_assignments(logits: np.ndarray, temperature: float, top_k: int, num_experts: int) -> dict:
    scaled_logits = logits / max(temperature, 1e-6)
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=-1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    top_indices = np.argsort(probs, axis=-1)[:, ::-1][:, :top_k]

    counts = np.zeros(num_experts, dtype=np.int64)
    for row in top_indices:
        for idx in row:
            counts[idx] += 1

    max_cnt = np.max(counts)
    mean_cnt = np.mean(counts)
    imbalance_ratio = float(max_cnt / mean_cnt) if mean_cnt > 0 else 1.0

    return {
        "probabilities": probs,
        "top_indices": top_indices,
        "expert_counts": counts,
        "imbalance_ratio": imbalance_ratio
    }
