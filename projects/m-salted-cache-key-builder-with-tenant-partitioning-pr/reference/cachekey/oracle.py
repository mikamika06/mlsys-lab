def infer_prefix_residency(ttft_samples, baseline_ttft_per_token, cache_hit_ttft):
    """Infer the length of cached prefix tokens from measured TTFT."""
    res = []
    for sample in ttft_samples:
        prompt_len = sample["prompt_len"]
        measured_ttft = sample["ttft"]
        saved_time = max(0.0, (prompt_len * baseline_ttft_per_token + cache_hit_ttft) - measured_ttft)
        tokens_cached = int(round(saved_time / baseline_ttft_per_token)) if baseline_ttft_per_token > 0 else 0
        tokens_cached = max(0, min(prompt_len, tokens_cached))
        res.append(tokens_cached)
    return res


def can_share_blocks(req_a, req_b, allow_cross_tenant=False):
    """Determine whether two requests are allowed to share cache blocks."""
    tenant_a = req_a.get("tenant_id")
    tenant_b = req_b.get("tenant_id")
    if not allow_cross_tenant and tenant_a != tenant_b:
        return False
    salt_a = req_a.get("salt", "")
    salt_b = req_b.get("salt", "")
    if salt_a != salt_b:
        return False
    return True
