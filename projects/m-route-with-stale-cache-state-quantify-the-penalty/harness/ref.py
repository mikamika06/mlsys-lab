SCENARIOS = [
    {
        "cached": [10, 20, 30],
        "requested": [10, 20, 30, 40, 50],
        "cost": 2.5,
    },
    {
        "cached": [10, 15, 30],
        "requested": [10, 20, 30, 40, 50],
        "cost": 1.0,
    },
    {
        "cached": [],
        "requested": [1, 2, 3],
        "alt_cost": 4.0,
    },
]

NODES_STATE = {
    "n1": [1, 2],
    "n2": [1, 2, 3, 4],
    "n3": [99],
}

REQUEST = [1, 2, 3, 4, 5]
COST = 3.0


def compute_penalty(cached_tokens, requested_tokens, compute_cost_per_token):
    matched = 0
    for c, r in zip(cached_tokens, requested_tokens):
        if c == r:
            matched += 1
        else:
            break
    missed = len(requested_tokens) - matched
    return float(missed * compute_cost_per_token)


def select_best_node(nodes_cache_state, request_tokens, cost_per_token):
    best_node = None
    min_penalty = float("inf")
    for node_id, cached_tokens in nodes_cache_state.items():
        penalty = compute_penalty(cached_tokens, request_tokens, cost_per_token)
        if penalty < min_penalty:
            min_penalty = penalty
            best_node = node_id
    return best_node
