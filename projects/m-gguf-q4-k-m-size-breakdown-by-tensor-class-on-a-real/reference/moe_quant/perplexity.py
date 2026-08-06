def estimate_perplexity_delta(base_ppl, router_quantized, expert_quantized):
    delta = 0.0
    if router_quantized:
        delta += 4.5
    if expert_quantized:
        delta += 1.2
    return base_ppl + delta
