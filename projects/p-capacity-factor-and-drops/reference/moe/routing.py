import numpy as np

def compute_drop_ratio(router_logits, capacity_factor):
    num_tokens, num_experts = router_logits.shape
    probs = np.exp(router_logits) / np.sum(np.exp(router_logits), axis=-1, keepdims=True)
    assignments = np.argmax(probs, axis=-1)
    counts = np.bincount(assignments, minlength=num_experts)
    capacity = int(np.ceil((num_tokens / num_experts) * capacity_factor))
    dropped = np.maximum(0, counts - capacity)
    return float(np.sum(dropped)) / float(num_tokens)

def fit_capacity_model(factors, drop_ratios):
    log_ratios = np.log(np.maximum(drop_ratios, 1e-6))
    poly = np.polyfit(factors, log_ratios, 1)
    return float(poly[0]), float(poly[1])

def zero_drop_routing(router_logits):
    num_tokens, num_experts = router_logits.shape
    probs = np.exp(router_logits) / np.sum(np.exp(router_logits), axis=-1, keepdims=True)
    assignments = np.argmax(probs, axis=-1)
    counts = np.bincount(assignments, minlength=num_experts)
    costs = counts.astype(float) / np.mean(counts)
    return float(np.max(costs))

def evaluate_quality(drop_ratio):
    return float(1.0 - 2.5 * drop_ratio)

def recommend_capacity_factor(workload_stats):
    max_skew = workload_stats.get("max_skew", 1.5)
    return float(1.0 + 0.5 * max_skew)

def peak_batch_routing(router_logits, capacity_factor):
    num_tokens, num_experts = router_logits.shape
    if num_tokens == 0:
        return 0.0
    return compute_drop_ratio(router_logits, capacity_factor)
