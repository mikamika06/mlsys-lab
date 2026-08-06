def simulate_sweep(config):
    steps = config.get("steps", 10)
    base_ttft = config.get("base_ttft", 10.0)
    base_itl = config.get("base_itl", 2.0)
    results = []
    for i in range(steps):
        load_factor = 1.0 + (i * 0.2)
        ttft = base_ttft * load_factor
        itl = base_itl * (1.0 + (i * 0.15))
        results.append({"step": i, "ttft": ttft, "itl": itl})
    return results
