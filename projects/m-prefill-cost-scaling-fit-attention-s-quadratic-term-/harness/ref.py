import numpy as np

def fit_scaling(lengths, times):
    p = np.polyfit(lengths, times, 2)
    return {"linear": float(p[1]), "quadratic": float(p[0])}

def diagnose_rope(config, inv_freq):
    dim = config["head_dim"]
    theta = config["rope_theta"]
    expected = 1.0 / (theta ** (np.arange(0, dim, 2) / dim))
    err = np.max(np.abs(expected - np.array(inv_freq)) / (expected + 1e-9))
    return bool(err > 1e-4)

def cheapest_config(model, gpus, ctx, budget):
    kv_bytes = model["num_layers"] * model["num_kv_heads"] * model["head_dim"] * 4
    req_gb = model["weights_gb"] + (kv_bytes * ctx) / (1024**3)

    best = None
    best_cost = float('inf')
    best_mem = -1

    for g in gpus:
        for n in range(1, g["count"] + 1):
            mem = n * g["mem_gb"]
            cost = n * g["cost_per_hr"]
            if mem >= req_gb and cost <= budget:
                if cost < best_cost or (cost == best_cost and mem > best_mem):
                    best = (g["name"], n)
                    best_cost = cost
                    best_mem = mem
    return best

SCALING_FIXTURES = [
    ([1024, 2048, 4096, 8192], [10.2, 21.0, 46.5, 110.0]),
    ([1000, 2000, 3000, 4000], [5.0, 11.0, 18.0, 27.0]),
]

ROPE_FIXTURES = [
    ({"head_dim": 128, "rope_theta": 10000.0}, (1.0 / (10000.0 ** (np.arange(0, 128, 2) / 128))).tolist()),
    ({"head_dim": 64, "rope_theta": 50000.0}, (1.0 / (10000.0 ** (np.arange(0, 64, 2) / 64))).tolist()),
]

MEMORY_FIXTURES = [
    (
        {"weights_gb": 14.0, "num_layers": 32, "num_kv_heads": 8, "head_dim": 128},
        [
            {"name": "T4", "mem_gb": 16, "cost_per_hr": 0.5, "count": 8},
            {"name": "A100", "mem_gb": 80, "cost_per_hr": 2.0, "count": 2},
        ],
        200000,
        5.0
    ),
    (
        {"weights_gb": 70.0, "num_layers": 80, "num_kv_heads": 8, "head_dim": 128},
        [
            {"name": "A10G", "mem_gb": 24, "cost_per_hr": 1.0, "count": 8},
            {"name": "A100", "mem_gb": 80, "cost_per_hr": 2.5, "count": 4},
        ],
        100000,
        6.0
    )
]
