CONFIGS = [
    {"target": "dense-7b", "draft": "dense-1b", "tokens_per_sec_base": 22.0, "accept_rates": {2: 0.75, 4: 0.65, 6: 0.50, 8: 0.35}},
    {"target": "dense-8b", "draft": "dense-1.5b", "tokens_per_sec_base": 18.5, "accept_rates": {2: 0.80, 4: 0.70, 6: 0.55, 8: 0.40}},
    {"target": "dense-14b", "draft": "dense-3b", "tokens_per_sec_base": 12.0, "accept_rates": {2: 0.82, 4: 0.72, 6: 0.58, 8: 0.42}},
]

def compute_speedup(base_tps, draft_tps, accept_rate, num_tokens):
    effective_tps = base_tps * (1.0 + accept_rate * (num_tokens / 4.0))
    return effective_tps / base_tps

def find_optimum(config):
    best_tokens = 2
    best_speedup = 0.0
    for n_tokens, rate in config["accept_rates"].items():
        s = compute_speedup(config["tokens_per_sec_base"], config["tokens_per_sec_base"] * 2.5, rate, n_tokens)
        if s > best_speedup:
            best_speedup = s
            best_tokens = n_tokens
    return {"optimal_tokens": best_tokens, "max_speedup": best_speedup}

def check_moe_regression(metrics):
    return metrics.get("moe_speedup", 1.0) < 1.0 or metrics.get("regression_detected", False)
