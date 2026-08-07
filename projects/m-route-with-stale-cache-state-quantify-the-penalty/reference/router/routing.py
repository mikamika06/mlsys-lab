from router.penalty import compute_penalty


def select_best_node(nodes_cache_state, request_tokens, cost_per_token):
    best_node = None
    min_penalty = float("inf")
    for node_id, cached_tokens in nodes_cache_state.items():
        penalty = compute_penalty(cached_tokens, request_tokens, cost_per_token)
        if penalty < min_penalty:
            min_penalty = penalty
            best_node = node_id
    return best_node
