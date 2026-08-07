import random

CONFIGS = [
    {"id": 0, "latency": 1.20, "shmem": 32768, "block_m": 64, "block_n": 64, "num_stages": 2},
    {"id": 1, "latency": 0.95, "shmem": 49152, "block_m": 128, "block_n": 64, "num_stages": 3},
    {"id": 2, "latency": 1.10, "shmem": 32768, "block_m": 64, "block_n": 64, "num_stages": 3},
    {"id": 3, "latency": 0.85, "shmem": 98304, "block_m": 128, "block_n": 128, "num_stages": 4},
    {"id": 4, "latency": 0.90, "shmem": 65536, "block_m": 128, "block_n": 64, "num_stages": 4},
    {"id": 5, "latency": 1.50, "shmem": 16384, "block_m": 32, "block_n": 32, "num_stages": 1}
]

def load_sweep(raw_data):
    out = []
    for line in raw_data.strip().split("\n"):
        if not line or line.startswith("#"):
            continue
        parts = line.split(",")
        out.append({
            "id": int(parts[0]),
            "latency": float(parts[1]),
            "shmem": int(parts[2]),
            "block_m": int(parts[3]),
            "block_n": int(parts[4]),
            "num_stages": int(parts[5])
        })
    return out

def compute_pareto(configs):
    frontier = []
    for c in configs:
        dominated = False
        for other in configs:
            if other["id"] == c["id"]:
                continue
            better_or_equal_latency = other["latency"] <= c["latency"]
            better_or_equal_shmem = other["shmem"] <= c["shmem"]
            strictly_better = (other["latency"] < c["latency"]) or (other["shmem"] < c["shmem"])
            if better_or_equal_latency and better_or_equal_shmem and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(c)
    return sorted(frontier, key=lambda x: (x["latency"], x["shmem"]))

def select_best(configs, max_shmem):
    valid = [c for c in configs if c["shmem"] <= max_shmem]
    if not valid:
        return None
    best = min(valid, key=lambda x: (x["latency"], x["shmem"]))
    return best["id"]

def raw_sweep_text():
    return "\n".join([
        f"{c['id']},{c['latency']},{c['shmem']},{c['block_m']},{c['block_n']},{c['num_stages']}"
        for c in CONFIGS
    ])
