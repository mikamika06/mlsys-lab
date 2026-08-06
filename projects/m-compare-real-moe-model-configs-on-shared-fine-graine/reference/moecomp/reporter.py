from moecomp.metrics import compute_costs

def compare_configs(cfg1, cfg2):
    c1 = compute_costs(cfg1)
    c2 = compute_costs(cfg2)
    ratio = c1["total_params"] / max(1, c2["total_params"])
    return {
        "config1_params": c1["total_params"],
        "config2_params": c2["total_params"],
        "ratio": ratio
    }
