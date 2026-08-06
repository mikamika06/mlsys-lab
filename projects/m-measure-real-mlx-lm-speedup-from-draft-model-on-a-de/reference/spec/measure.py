def measure_speedup(config, num_draft_tokens=4, override_rate=None):
    base = config["tokens_per_sec_base"]
    rate = override_rate if override_rate is not None else config["accept_rates"].get(num_draft_tokens, 0.5)
    effective_tps = base * (1.0 + rate * (num_draft_tokens / 4.0))
    return effective_tps / base
