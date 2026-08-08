def optimal_eagle_config(kv_budget_bytes: int, candidate_configs: list) -> dict:
    best_cfg = None
    best_score = -1.0
    for cfg in candidate_configs:
        req_bytes = cfg["kv_per_token_bytes"] * cfg["max_context_len"] * cfg["max_batch_size"]
        if req_bytes <= kv_budget_bytes:
            score = cfg["draft_tokens"] * cfg["acceptance_rate"]
            if score > best_score:
                best_score = score
                best_cfg = cfg
    return best_cfg or candidate_configs[0]
