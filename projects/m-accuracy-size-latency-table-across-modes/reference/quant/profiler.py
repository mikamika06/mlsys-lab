import numpy as np


def profile_modes(model_spec, dataset):
    modes = ["fp32", "fp16", "int8_fallback", "int8_full"]
    base_size = model_spec["base_size"]
    base_acc = model_spec["base_accuracy"]
    base_lat = model_spec["base_latency"]
    size_factors = {"fp32": 1.0, "fp16": 0.5, "int8_fallback": 0.3, "int8_full": 0.25}
    acc_drops = {"fp32": 0.0, "fp16": 0.001, "int8_fallback": 0.008, "int8_full": 0.025}
    lat_factors = {"fp32": 1.0, "fp16": 0.85, "int8_fallback": 0.7, "int8_full": 0.6}
    profiles = []
    for mode in modes:
        sz = int(base_size * size_factors[mode])
        acc = float(max(0.0, base_acc - acc_drops[mode]))
        lat = float(base_lat * lat_factors[mode])
        size_ratio = float(sz / base_size)
        profiles.append({
            "mode": mode,
            "size": sz,
            "accuracy": acc,
            "latency": lat,
            "size_ratio": size_ratio
        })
    return profiles
