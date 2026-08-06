import heapq
import numpy as np


def rebalance_greedy_redundant(expert_loads, num_ranks, num_extra_replicas):
    loads = np.asarray(expert_loads, dtype=np.float64)
    num_experts = len(loads)

    replicas = [1] * num_experts
    expert_load_per_replica = list(loads.copy())

    expert_placements = {e: [] for e in range(num_experts)}

    rank_heap = [(0.0, r) for r in range(num_ranks)]
    heapq.heapify(rank_heap)

    sorted_experts = sorted(range(num_experts), key=lambda x: loads[x], reverse=True)

    for e in sorted_experts:
        l, r = heapq.heappop(rank_heap)
        expert_placements[e].append(r)
        heapq.heappush(rank_heap, (l + loads[e], r))

    for _ in range(num_extra_replicas):
        best_e = max(range(num_experts), key=lambda e: expert_load_per_replica[e])
        replicas[best_e] += 1
        expert_load_per_replica[best_e] = loads[best_e] / replicas[best_e]

        assigned_ranks = set(expert_placements[best_e])
        candidate_ranks = [r for r in range(num_ranks) if r not in assigned_ranks]

        if not candidate_ranks:
            candidate_ranks = list(range(num_ranks))

        rank_loads = np.zeros(num_ranks, dtype=np.float64)
        for e, r_list in expert_placements.items():
            cost = loads[e] / len(r_list)
            for r in r_list:
                rank_loads[r] += cost

        target_rank = min(candidate_ranks, key=lambda r: rank_loads[r])
        expert_placements[best_e].append(target_rank)

    final_layout = [expert_placements[e] for e in range(num_experts)]
    final_rank_loads = np.zeros(num_ranks, dtype=np.float64)
    for e, r_list in enumerate(final_layout):
        cost = loads[e] / len(r_list)
        for r in r_list:
            final_rank_loads[r] += cost

    return {
        "layout": final_layout,
        "max_rank_load": float(np.max(final_rank_loads)),
        "rank_loads": final_rank_loads.tolist(),
        "replicas": replicas
    }
