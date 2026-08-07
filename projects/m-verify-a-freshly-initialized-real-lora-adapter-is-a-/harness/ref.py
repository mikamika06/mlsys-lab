import numpy as np

CONFIGS = [
    {"in_features": 64, "out_features": 128, "rank": 4, "alpha": 8},
    {"in_features": 128, "out_features": 256, "rank": 8, "alpha": 16},
    {"in_features": 32, "out_features": 64, "rank": 2, "alpha": 4},
]

def make_layer(cfg, seed=42):
    rng = np.random.default_rng(seed)
    w = rng.normal(0, 0.02, (cfg["out_features"], cfg["in_features"]))
    a = rng.normal(0, 0.02, (cfg["rank"], cfg["in_features"]))
    b = np.zeros((cfg["out_features"], cfg["rank"]))
    return w, a, b
