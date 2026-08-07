import random

def plan_engine(config, profile, workspace_limit):
    max_s = profile["max_s"]
    opt_s = profile["opt_s"]
    max_ws = 0
    total_lat = 0.0

    for layer in config["layers"]:
        best_lat = float('inf')
        chosen_ws = 0
        for t in layer["tactics"]:
            ws = t["base_ws"] + t["ws_factor"] * max_s
            if ws <= workspace_limit:
                lat = t["base_lat"] + t["lat_factor"] * opt_s
                if lat < best_lat:
                    best_lat = lat
                    chosen_ws = ws
        if best_lat == float('inf'):
            return float('inf'), float('inf')
        total_lat += best_lat
        max_ws = max(max_ws, chosen_ws)

    return config["weights_memory"] + max_ws, total_lat

def sweep_workspace(config, profile, device_memory, limits):
    best_idx = -1
    best_lat = float('inf')

    for i, limit in enumerate(limits):
        mem, lat = plan_engine(config, profile, limit)
        if mem <= device_memory and lat < best_lat:
            best_lat = lat
            best_idx = i

    return best_idx

def _make_scenario(seed):
    rng = random.Random(seed)
    profile = {"max_s": rng.randint(512, 1024), "opt_s": rng.randint(32, 128)}
    config = {"weights_memory": rng.randint(1000, 5000), "layers": []}

    for _ in range(rng.randint(3, 8)):
        tactics = []
        for _ in range(rng.randint(2, 5)):
            tactics.append({
                "base_ws": rng.randint(100, 1000),
                "ws_factor": rng.randint(1, 10),
                "base_lat": rng.randint(10, 50),
                "lat_factor": rng.random() * 0.5
            })
        config["layers"].append({"tactics": tactics})

    device_memory = config["weights_memory"] + rng.randint(2000, 10000)
    limits = [rng.randint(500, 5000) for _ in range(5)]
    limits.sort()
    return config, profile, device_memory, limits

SCENARIOS = [_make_scenario(i) for i in range(20)]
