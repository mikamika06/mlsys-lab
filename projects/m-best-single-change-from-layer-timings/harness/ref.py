import json


def generate_fixtures():
    raw = {
        "deviceMemorySize": 1048576,
        "layers": [
            {"name": "Conv_0", "type": "Convolution"},
            {"name": "Relu_0", "type": "Activation"},
            {"name": "Conv_1", "type": "Convolution"},
            {"name": "Relu_1", "type": "Activation"}
        ]
    }
    
    simp = {
        "deviceMemorySize": 1048576,
        "layers": [
            {"name": "Conv_0+Relu_0", "type": "Convolution"},
            {"name": "Conv_1+Relu_1", "type": "Convolution"}
        ]
    }
    
    prof = {
        "layers": [
            {"name": "Conv_0+Relu_0", "timeMs": 12.5},
            {"name": "Conv_1+Relu_1", "timeMs": 8.2},
            {"name": "Dense_0", "timeMs": 15.0}
        ]
    }
    
    candidates = [
        {"Conv_0+Relu_0": 10.0},
        {"Dense_0": 9.0},
        {"Conv_0+Relu_0": 11.0, "Conv_1+Relu_1": 4.0},
        {"Dense_0": 14.0}
    ]
    
    return raw, simp, prof, candidates


def analyze_engine(raw_json_path, simp_json_path):
    with open(raw_json_path, "r") as f:
        raw = json.load(f)
    with open(simp_json_path, "r") as f:
        simp = json.load(f)
    mem = raw.get("deviceMemorySize", 0)
    return mem, len(raw.get("layers", [])), len(simp.get("layers", []))


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
