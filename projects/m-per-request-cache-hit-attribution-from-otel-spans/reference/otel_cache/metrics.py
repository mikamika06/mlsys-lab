def compute_efficiency(attribution):
    total_hit = sum(v["hit_tokens"] for v in attribution.values())
    total_tokens = sum(v["total_tokens"] for v in attribution.values())
    if total_tokens == 0:
        return 0.0
    return float(total_hit) / float(total_tokens)
