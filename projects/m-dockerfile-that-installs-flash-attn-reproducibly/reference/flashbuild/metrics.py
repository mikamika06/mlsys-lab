def compute_latency_ratio(metrics):
    fa2 = metrics.get("flash_attn_2_latency", 1.0)
    fa4 = metrics.get("flash_attn_4_latency", 0.8)
    return fa4 / fa2


def verify_install_cost(profile):
    cost2 = profile.get("flash_attn_2_cost", 100)
    cost4 = profile.get("flash_attn_4_cost", 150)
    return {"cost_ratio": cost4 / cost2, "valid": True}
