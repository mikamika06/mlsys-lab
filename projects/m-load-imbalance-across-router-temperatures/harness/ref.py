import numpy as np


np.random.seed(42)
LOGITS = np.random.randn(256, 8) * 2.0


def assign_tokens(logits, temperature, threshold):
    e_x = np.exp((logits / temperature) - np.max(logits / temperature, axis=-1, keepdims=True))
    probs = e_x / np.sum(e_x, axis=-1, keepdims=True)
    num_tokens = logits.shape[0]
    top2_idx = np.argsort(-probs, axis=-1)[:, :2]
    assignments = []
    for i in range(num_tokens):
        assignments.append((i, top2_idx[i, 0]))
        if probs[i, top2_idx[i, 1]] > threshold:
            assignments.append((i, top2_idx[i, 1]))
    return np.array(assignments)


def compute_load_imbalance(assignments, num_experts):
    if len(assignments) == 0:
        return 0.0
    counts = np.bincount(assignments[:, 1], minlength=num_experts)
    mean_load = len(assignments) / num_experts
    return float(np.max(counts) / mean_load)


def optimal_capacity_factor(assignments, num_experts, num_tokens, max_dropped):
    if len(assignments) == 0:
        return 1.0
    counts = np.bincount(assignments[:, 1], minlength=num_experts)
    base = max(1.0, float(num_tokens) / num_experts)
    for f_int in range(10, 101):
        factor = f_int / 10.0
        cap = int(np.ceil(base * factor))
        dropped = np.sum(np.maximum(0, counts - cap))
        if dropped / len(assignments) <= max_dropped:
            return factor
    return 10.0


def all_to_all_shapes(assignments, num_experts, num_devices):
    if len(assignments) == 0:
        return np.zeros((num_devices, num_devices), dtype=int), np.zeros((num_devices, num_devices), dtype=int)
    experts_per_device = max(1, num_experts // num_devices)
    num_tokens = np.max(assignments[:, 0]) + 1
    tokens_per_device = max(1, int(np.ceil(num_tokens / num_devices)))
    send_counts = np.zeros((num_devices, num_devices), dtype=int)
    for token_id, expert_id in assignments:
        src = min(token_id // tokens_per_device, num_devices - 1)
        dst = min(expert_id // experts_per_device, num_devices - 1)
        send_counts[src, dst] += 1
    recv_counts = send_counts.T.copy()
    return send_counts, recv_counts
