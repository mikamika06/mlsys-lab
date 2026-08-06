import numpy as np


def normalize_and_repair_routing(router_weights, top_k, fallback_expert=0):
    weights = np.array(router_weights, dtype=np.float64, copy=True)
    num_tokens, num_experts = weights.shape

    repaired_flags = np.zeros(num_tokens, dtype=bool)

    for i in range(num_tokens):
        row_sum = np.sum(weights[i])
        if row_sum == 0.0:
            weights[i, :] = 0.0
            weights[i, fallback_expert] = 1.0
            repaired_flags[i] = True
        else:
            top_indices = np.argsort(-weights[i])[:top_k]
            mask = np.zeros(num_experts, dtype=bool)
            mask[top_indices] = True
            weights[i, ~mask] = 0.0
            new_sum = np.sum(weights[i])
            if new_sum > 0.0:
                weights[i] /= new_sum
            else:
                weights[i, :] = 0.0
                weights[i, fallback_expert] = 1.0
                repaired_flags[i] = True

    return {
        "repaired_weights": weights,
        "repaired_rows": np.where(repaired_flags)[0].tolist()
    }
