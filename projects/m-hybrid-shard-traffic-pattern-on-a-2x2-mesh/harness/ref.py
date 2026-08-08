CONFIGS = [
    {"mesh_shape": (2, 2), "shard_config": {"intra_group": 2, "inter_group": 2}},
    {"mesh_shape": (2, 2), "shard_config": {"intra_group": 1, "inter_group": 4}},
    {"mesh_shape": (2, 2), "shard_config": {"intra_group": 4, "inter_group": 1}},
]

POLICIES = [
    ({"a": 500, "b": 2000}, 1000),
    ({"x": 100, "y": 200}, 500),
    ({"m1": 5000, "m2": 50}, 1000),
]

STRATEGIES = [
    (["FULL_SHARD", "HYBRID_SHARD", "NO_SHARD"], 1500, {"FULL_SHARD": 1000, "HYBRID_SHARD": 2000, "NO_SHARD": 4000}),
    (["FULL_SHARD", "HYBRID_SHARD"], 500, {"FULL_SHARD": 1000, "HYBRID_SHARD": 800}),
    (["HYBRID_SHARD", "NO_SHARD"], 3000, {"HYBRID_SHARD": 2500, "NO_SHARD": 5000}),
]


def compute_traffic(mesh_shape, shard_config):
    rows, cols = mesh_shape
    total_ranks = rows * cols
    traffic = {}
    for r in range(total_ranks):
        row_idx, col_idx = divmod(r, cols)
        peers = []
        for other in range(total_ranks):
            o_row, o_col = divmod(other, cols)
            if o_row == row_idx:
                peers.append((other, "intra"))
            elif o_col == col_idx:
                peers.append((other, "inter"))
            else:
                peers.append((other, "cross"))
        traffic[r] = peers
    return traffic


def diagnose_policy(module_tree, min_size):
    issues = []
    for name, size in module_tree.items():
        if size < min_size:
            issues.append({"module": name, "size": size, "error": "too_small"})
    return issues


def select_strategy(strategies, memory_budget, model_memory):
    valid = []
    for s in strategies:
        req = model_memory.get(s, float("inf"))
        if req <= memory_budget:
            valid.append((s, req))
    if not valid:
        return min(strategies, key=lambda x: model_memory.get(x, float("inf")))
    return min(valid, key=lambda x: x[1])[0]
