from eplb.redundant import rebalance_greedy_redundant


def min_redundant_replicas(expert_loads, num_ranks, target_max_load):
    extra = 0
    max_limit = len(expert_loads) * num_ranks
    while extra <= max_limit:
        res = rebalance_greedy_redundant(expert_loads, num_ranks, extra)
        if res["max_rank_load"] <= target_max_load + 1e-9:
            return extra
        extra += 1
    return extra
