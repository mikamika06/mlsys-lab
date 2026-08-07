def infer_prefix_residency(ttft_samples, baseline_ttft_per_token, cache_hit_ttft):
    """Infer the length of cached prefix tokens from measured TTFT."""
    raise NotImplementedError


def can_share_blocks(req_a, req_b, allow_cross_tenant=False):
    """Determine whether two requests are allowed to share cache blocks."""
    raise NotImplementedError
