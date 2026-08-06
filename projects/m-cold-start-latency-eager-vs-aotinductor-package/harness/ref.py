import numpy as np


def generate_configs():
    np.random.seed(42)
    configs = []
    for i in range(4):
        configs.append({
            "model_id": f"model_{i}",
            "hidden_dim": int(32 * (i + 1)),
            "num_layers": int(2 + i),
            "supports_export": bool(i % 2 == 0),
            "cold_eager": float(10.0 + i * 2.5),
            "cold_aot": float(4.0 + i * 1.2),
            "warm_eager": float(1.0 + i * 0.2),
            "warm_aot": float(0.5 + i * 0.1),
            "min_seq": 1,
            "max_seq": 512,
        })
    return configs


CONFIGS = generate_configs()


def verify_export(config):
    return config["supports_export"]


def compute_latency_ratio(config):
    eager_cost = config["cold_eager"] / max(config["warm_eager"], 1e-5)
    aot_cost = config["cold_aot"] / max(config["warm_aot"], 1e-5)
    return float(eager_cost / max(aot_cost, 1e-5))


def derive_assertions(config):
    return {
        "min_seq": config["min_seq"],
        "max_seq": config["max_seq"],
        "hidden_dim": config["hidden_dim"],
        "valid": True
    }
