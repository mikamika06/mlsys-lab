import numpy as np

PROFILES = [
    {"kernel_ms": 1.2, "host_ms": 0.1, "synced": True},
    {"kernel_ms": 0.05, "host_ms": 5.0, "synced": False},
    {"kernel_ms": 2.5, "host_ms": 2.5, "synced": True},
]

REGIMES = [
    {"name": "heavy_kernel_low_host", "kernel_ms": 10.0, "host_ms": 0.1, "regime_id": 0},
    {"name": "light_kernel_heavy_host", "kernel_ms": 0.02, "host_ms": 8.0, "regime_id": 1},
    {"name": "balanced_sync", "kernel_ms": 2.0, "host_ms": 2.0, "regime_id": 2},
    {"name": "micro_kernel_flooded_queue", "kernel_ms": 0.005, "host_ms": 12.0, "regime_id": 3},
]

def simulate_events(profile):
    k = profile["kernel_ms"]
    h = profile["host_ms"]
    if profile["synced"]:
        elapsed = k + np.random.normal(0, 0.001)
    else:
        elapsed = h + k + np.random.normal(0, 0.01)
    return float(max(0.001, elapsed))

def rank_regimes(profiles):
    ranked = []
    for p in profiles:
        k = p["kernel_ms"]
        h = p["host_ms"]
        if k > h * 2:
            rid = 0
        elif h > k * 2 and k < 0.1:
            rid = 3
        elif h > k:
            rid = 1
        else:
            rid = 2
        ranked.append({"name": p["name"], "regime_id": rid})
    return ranked
