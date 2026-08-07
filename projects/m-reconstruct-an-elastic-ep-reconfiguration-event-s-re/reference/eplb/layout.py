import numpy as np


def compute_layout(num_experts, num_ranks, expert_weights, max_replicas_per_rank):
    weights = np.array(expert_weights, dtype=np.float64)
    if weights.sum() == 0:
        weights = np.ones_like(weights)

    total_replicas = min(num_experts, num_ranks * max_replicas_per_rank)
    replica_counts = np.ones(num_experts, dtype=np.int64)
    remaining = total_replicas - num_experts

    if remaining > 0:
        shares = weights / weights.sum() * remaining
        extra = np.floor(shares).astype(np.int64)
        replica_counts += extra
        rem_extra = remaining - extra.sum()
        if rem_extra > 0:
            fracs = shares - extra
            sorted_indices = np.argsort(-fracs)
            for i in range(rem_extra):
                replica_counts[sorted_indices[i]] += 1

    layout = {r: [] for r in range(num_ranks)}
    current_rank = 0
    for exp_id in range(num_experts):
        reps = replica_counts[exp_id]
        for _ in range(reps):
            for _ in range(num_ranks):
                if len(layout[current_rank]) < max_replicas_per_rank:
                    layout[current_rank].append(exp_id)
                    current_rank = (current_rank + 1) % num_ranks
                    break
                current_rank = (current_rank + 1) % num_ranks
    for r in range(num_ranks):
        layout[r] = sorted(layout[r])
    return layout
