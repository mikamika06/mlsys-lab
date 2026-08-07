import random

def simulate_generation(config, guided=False, seed=42):
    rng = random.Random(seed)
    base_tpot = 15.0 if not guided else 45.0
    factor = 1.0 + (0.05 if config["schema_complexity"] == "low" else 0.2 if config["schema_complexity"] == "medium" else 0.4)
    if guided:
        factor *= 1.5
    latencies = []
    for _ in range(config["output_tokens"]):
        noise = rng.uniform(-1.0, 1.0)
        val = max(5.0, base_tpot * factor + noise)
        latencies.append(val)
    return latencies
