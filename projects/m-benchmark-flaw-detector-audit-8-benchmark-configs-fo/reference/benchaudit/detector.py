def audit_benchmark_configs(configs: list[dict]) -> list[dict]:
    """Audit benchmark configurations for methodology flaws."""
    results = []
    for cfg in configs:
        cfg_id = cfg.get("config_id", "")
        flaws = []
        if cfg.get("num_warmup_requests", 0) <= 0:
            flaws.append("missing_warmup")
        if cfg.get("ignore_eos", False) is True:
            flaws.append("ignore_eos_bias")
        min_p = cfg.get("min_prompt_len")
        max_p = cfg.get("max_prompt_len")
        min_o = cfg.get("min_output_len")
        max_o = cfg.get("max_output_len")
        if (min_p is not None and max_p is not None and min_p == max_p) or \
           (min_o is not None and max_o is not None and min_o == max_o):
            flaws.append("length_bias")
        results.append({
            "config_id": cfg_id,
            "flaws": sorted(flaws)
        })
    return results
