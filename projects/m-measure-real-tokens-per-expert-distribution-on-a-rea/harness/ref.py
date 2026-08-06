import numpy as np


def generate_test_logits(num_layers=4, batch_seq=256, num_experts=16, seed=42):
    rng = np.random.RandomState(seed)
    layers = []
    for _ in range(num_layers):
        logits = rng.randn(batch_seq, num_experts)
        logits[:, 0] += 3.0
        logits[:, 1] += 1.5
        layers.append(logits)
    return layers


def compute_expert_load(gate_logits, top_k):
    batch_seq, num_experts = gate_logits.shape
    top_k_indices = np.argsort(gate_logits, axis=-1)[:, -top_k:]
    return np.bincount(top_k_indices.ravel(), minlength=num_experts)


def analyze_imbalance(expert_counts, num_experts):
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


def simulate_loss_free_routing(layer_logits, top_k):
    counts_by_layer = []
    for logits in layer_logits:
        counts = compute_expert_load(logits, top_k)
        counts_by_layer.append(counts)
    return np.array(counts_by_layer, dtype=np.int64)


def measure_sparsity_pathology(layer_logits_dict, top_k_list):
    results = {}
    num_experts = layer_logits_dict[0].shape[1]
    for k in top_k_list:
        counts = simulate_loss_free_routing(layer_logits_dict, k)
        agg_counts = np.sum(counts, axis=0)
        metrics = analyze_imbalance(agg_counts, num_experts)
        results[k] = metrics
    return results
