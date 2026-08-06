import numpy as np
from moe_dist.metrics import analyze_imbalance, compute_expert_load


def simulate_loss_free_routing(layer_logits, top_k):
    """Route tokens without load balancing loss and return per-layer expert counts."""
    num_layers = len(layer_logits)
    num_experts = layer_logits[0].shape[1]
    counts_by_layer = []
    for logits in layer_logits:
        counts = compute_expert_load(logits, top_k)
        counts_by_layer.append(counts)
    return np.array(counts_by_layer, dtype=np.int64)


def measure_sparsity_pathology(layer_logits_dict, top_k_list):
    """Compare load metrics across different top_k settings to demonstrate imbalance."""
    results = {}
    num_experts = layer_logits_dict[0].shape[1]
    for k in top_k_list:
        counts = simulate_loss_free_routing(layer_logits_dict, k)
        agg_counts = np.sum(counts, axis=0)
        metrics = analyze_imbalance(agg_counts, num_experts)
        results[k] = metrics
    return results
