import random

def generate_logs():
    random.seed(42)
    logs = []
    modes = ["compact", "balanced", "explicit", "none"]
    for i in range(10):
        mode = modes[i % len(modes)]
        latencies = [random.randint(10, 100) + (20 if mode == "none" else 0) for _ in range(5)]
        logs.append({
            "run_id": i,
            "affinity_mode": mode,
            "latencies": latencies
        })
    return logs

def compute_latency_ratio(logs):
    baselines = [l["latencies"] for l in logs if l["affinity_mode"] == "balanced"]
    base_avg = sum(sum(b) / len(b) for b in baselines) / len(baselines) if baselines else 1.0
    ratios = {}
    for l in logs:
        avg = sum(l["latencies"]) / len(l["latencies"])
        ratios[l["run_id"]] = avg / base_avg
    return ratios

def classify_state(config):
    threads = config.get("threads", 1)
    cores = config.get("cores", 1)
    if threads > cores:
        return "oversubscribed"
    elif threads < cores:
        return "under-pinned"
    else:
        return "optimal"

CONFIGS = [
    {"threads": 8, "cores": 8},
    {"threads": 16, "cores": 8},
    {"threads": 4, "cores": 8},
    {"threads": 32, "cores": 32}
]
