import numpy as np


def generate_mock_runs(seed=42):
    rng = np.random.default_rng(seed)
    base_latency = 1.25
    noise = rng.normal(0, 0.05, size=100)
    return (base_latency + noise).tolist()


CONFIGS = [
    {"batch": 2, "seqlen": 2048, "heads": 32, "dim": 128},
    {"batch": 4, "seqlen": 4096, "heads": 16, "dim": 64},
    {"batch": 1, "seqlen": 8192, "heads": 8, "dim": 256},
]
