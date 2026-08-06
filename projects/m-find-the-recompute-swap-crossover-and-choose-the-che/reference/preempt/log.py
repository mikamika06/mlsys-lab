def calculate_wasted_compute(events: list) -> float:
    total_wasted_flops = 0.0
    for ev in events:
        tokens = ev["tokens_processed"]
        params_b = ev["model_params_b"]
        wasted_flops = 2.0 * params_b * 1e9 * tokens
        total_wasted_flops += wasted_flops
    return total_wasted_flops
