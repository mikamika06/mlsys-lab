def summarize_performance(bias, comparison):
    tier = "high_bias" if bias > 0.05 else "stable"
    return {"tier": tier, "bias": bias, "comparison": comparison}
