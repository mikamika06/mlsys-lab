import numpy as np

CONFIGS = [
    {
        "hidden_dim": 64,
        "num_experts": 4,
        "top_k": 2,
        "intermediate_dim": 128,
        "num_layers": 1,
    },
    {
        "hidden_dim": 128,
        "num_experts": 8,
        "top_k": 2,
        "intermediate_dim": 256,
        "num_layers": 2,
    },
]


def generate_router_weights(seed=42):
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(8, 128))
    w[3, :] = 0.0
    return w


def count_parameters_ref(config):
    h = config["hidden_dim"]
    e = config["num_experts"]
    k = config["top_k"]
    i = config["intermediate_dim"]
    l = config.get("num_layers", 1)
    router = h * e
    expert = 2 * h * i
    total = l * (router + e * expert)
    active = l * (router + k * expert)
    return {"total_parameters": int(total), "active_parameters": int(active)}
