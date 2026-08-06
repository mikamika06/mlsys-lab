def estimate_install_cost(config):
    v = config.get("flash_version", "2.6.3")
    arch = config.get("gpu_arch", "89")
    base = 140.0 if v.startswith("4") else 90.0
    mult = 1.25 if arch == "90" else 1.0
    return base * mult

def latency_ratio(config_a, config_b):
    return estimate_install_cost(config_a) / (estimate_install_cost(config_b) + 1e-6)
