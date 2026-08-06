import numpy as np

def generate_test_data(seed=42):
    rng = np.random.default_rng(seed)
    num_tokens = 64
    num_experts = 4
    top_k = 2
    tokens = rng.normal(size=(num_tokens, 8))
    logits = rng.normal(size=(num_tokens, num_experts))
    return tokens, logits, top_k

def compute_dropless_output(tokens, logits, top_k):
    num_tokens, num_experts = logits.shape
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    top_indices = np.argsort(probs, axis=-1)[:, -top_k:]
    top_probs = np.take_along_axis(probs, top_indices, axis=-1)
    top_probs = top_probs / np.sum(top_probs, axis=-1, keepdims=True)

    out = np.zeros_like(tokens)
    expert_outputs = [tokens * 1.1, tokens * 0.9, tokens * 1.2, tokens * 0.8]

    for i in range(num_tokens):
        for k_idx in range(top_k):
            exp_id = top_indices[i, k_idx]
            p = top_probs[i, k_idx]
            out[i] += p * expert_outputs[exp_id][i]
    return out

def compute_capacity_limited_output(tokens, logits, top_k, capacity_factor):
    num_tokens, num_experts = logits.shape
    capacity = int(np.ceil(capacity_factor * num_tokens * top_k / num_experts))

    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    top_indices = np.argsort(probs, axis=-1)[:, -top_k:]
    top_probs = np.take_along_axis(probs, top_indices, axis=-1)

    expert_counts = np.zeros(num_experts, dtype=int)
    expert_outputs = [tokens * 1.1, tokens * 0.9, tokens * 1.2, tokens * 0.8]
    out = np.zeros_like(tokens)

    for i in range(num_tokens):
        for k_idx in range(top_k):
            exp_id = top_indices[i, k_idx]
            if expert_counts[exp_id] < capacity:
                expert_counts[exp_id] += 1
                p = top_probs[i, k_idx]
                out[i] += p * expert_outputs[exp_id][i]
    return out

def find_min_capacity_factor(tokens, logits, top_k, max_drop_rate):
    factors = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    num_tokens, num_experts = logits.shape
    total_assignments = num_tokens * top_k

    for cf in factors:
        capacity = int(np.ceil(cf * total_assignments / num_experts))
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        top_indices = np.argsort(probs, axis=-1)[:, -top_k:]

        expert_counts = np.zeros(num_experts, dtype=int)
        dropped = 0
        for i in range(num_tokens):
            for k_idx in range(top_k):
                exp_id = top_indices[i, k_idx]
                if expert_counts[exp_id] < capacity:
                    expert_counts[exp_id] += 1
                else:
                    dropped += 1
        drop_rate = dropped / total_assignments
        if drop_rate <= max_drop_rate:
            return cf
    return 2.0
