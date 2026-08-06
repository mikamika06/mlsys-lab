def compute_wasted_compute(preemption_log, model_config, system_config):
    total_wasted = 0.0
    for entry in preemption_log:
        tokens = entry["tokens_recomputed"]
        total_wasted += tokens * system_config["flops_per_token"]
    return total_wasted
