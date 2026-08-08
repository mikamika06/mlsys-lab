def diagnose_policy(layer_configs, wrap_policy):
    issues = []
    for cfg in layer_configs:
        size = cfg.get("size", 0)
        min_size = wrap_policy.get("min_num_params", 0)
        if size < min_size and cfg.get("wrapper") == "heavy":
            issues.append(cfg["name"])
    return {"misconfigured": issues, "valid": len(issues) == 0}
