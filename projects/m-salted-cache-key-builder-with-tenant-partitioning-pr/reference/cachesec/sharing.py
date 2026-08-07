def can_share_blocks(req_a: dict, req_b: dict) -> bool:
    if req_a.get("tenant_id") != req_b.get("tenant_id"):
        return False
    tokens_a = req_a.get("prefix_tokens", [])
    tokens_b = req_b.get("prefix_tokens", [])
    min_len = min(len(tokens_a), len(tokens_b))
    return tokens_a[:min_len] == tokens_b[:min_len]
