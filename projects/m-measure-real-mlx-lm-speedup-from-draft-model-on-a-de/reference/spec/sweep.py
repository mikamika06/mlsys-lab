from spec.measure import measure_speedup


def sweep_draft_tokens(config):
    results = {}
    best_tokens = 2
    best_speedup = 0.0
    for n_tokens, rate in config["accept_rates"].items():
        s = measure_speedup(config, num_draft_tokens=n_tokens, override_rate=rate)
        results[n_tokens] = s
        if s > best_speedup:
            best_speedup = s
            best_tokens = n_tokens
    return {"optimal_tokens": best_tokens, "max_speedup": best_speedup, "sweep_results": results}
