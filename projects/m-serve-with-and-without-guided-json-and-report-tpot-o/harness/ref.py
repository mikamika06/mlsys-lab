import random

CONFIGS = [
    {"schema_complexity": "low", "vocab_size": 32000, "output_tokens": 64},
    {"schema_complexity": "medium", "vocab_size": 32000, "output_tokens": 128},
    {"schema_complexity": "high", "vocab_size": 32000, "output_tokens": 256},
]

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

def compute_tpot_overhead(config, seed=42):
    t_unconstrained = simulate_generation(config, guided=False, seed=seed)
    t_guided = simulate_generation(config, guided=True, seed=seed)
    mean_unconstrained = sum(t_unconstrained) / len(t_unconstrained)
    mean_guided = sum(t_guided) / len(t_guided)
    ratio = mean_guided / mean_unconstrained
    return {
        "mean_unconstrained_tpot": mean_unconstrained,
        "mean_guided_tpot": mean_guided,
        "latency_ratio": ratio
    }

def generate_report(config, seed=42):
    res = compute_tpot_overhead(config, seed=seed)
    return f"Schema: {config['schema_complexity']} | Unconstrained TPOT: {res['mean_unconstrained_tpot']:.2f}ms | Guided TPOT: {res['mean_guided_tpot']:.2f}ms | Ratio: {res['latency_ratio']:.2f}x"
