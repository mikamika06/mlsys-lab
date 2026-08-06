import numpy as np


def compute_expert_load(gate_logits, top_k):
    """Compute tokens per expert and load distribution statistics."""
    raise NotImplementedError


def analyze_imbalance(expert_counts, num_experts):
    """Compute coefficient of variation, peak ratio, and starvation count."""
    raise NotImplementedError
