def evaluate_cold_start(config):
    eager_cost = config["cold_eager"] / max(config["warm_eager"], 1e-5)
    aot_cost = config["cold_aot"] / max(config["warm_aot"], 1e-5)
    return float(eager_cost / max(aot_cost, 1e-5))
