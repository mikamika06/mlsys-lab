import json


def analyze_engine(raw_json_path, simp_json_path):
    with open(raw_json_path, "r") as f:
        raw = json.load(f)
    with open(simp_json_path, "r") as f:
        simp = json.load(f)

    mem = raw.get("deviceMemorySize", 0)
    raw_count = len(raw.get("layers", []))
    simp_count = len(simp.get("layers", []))

    return mem, raw_count, simp_count


def best_single_change(profile_path, candidates):
    with open(profile_path, "r") as f:
        profile = json.load(f)

    layer_times = {layer["name"]: layer["timeMs"] for layer in profile.get("layers", [])}
    base_time = sum(layer_times.values())

    best_idx = -1
    best_time = float("inf")

    for i, cand in enumerate(candidates):
        new_total = base_time
        for target, new_time in cand.items():
            if target in layer_times:
                new_total += (new_time - layer_times[target])
        
        if new_total < best_time:
            best_time = new_total
            best_idx = i

    return best_idx
