import numpy as np


def compute_expert_load(gate_logits, top_k):
    """Compute tokens per expert and load distribution statistics."""
    batch_seq, num_experts = gate_logits.shape
    top_k_indices = np.argsort(gate_logits, axis=-1)[:, -top_k:]
    expert_counts = np.bincount(top_k_indices.ravel(), minlength=num_experts)
    return expert_counts


def analyze_imbalance(expert_counts, num_experts):
    """Compute coefficient of variation, peak ratio, and starvation count."""
    expert_counts = np.asarray(expert_counts, dtype=np.float64)
    total_tokens = np.sum(expert_counts)
    mean_load = total_tokens / num_experts
    std_load = np.std(expert_counts, ddof=0)
    cv = std_load / mean_load if mean_load > 0 else 0.0
    peak_ratio = np.max(expert_counts) / mean_load if mean_load > 0 else 0.0
    starved_count = int(np.sum(expert_counts == 0))
    return {
        "total_assignments": int(total_tokens),
        "cv": float(cv),
        "peak_ratio": float(peak_ratio),
        "starved_experts": starved_count,
    }
