import numpy as np

np.random.seed(42)

def make_layers():
    return [
        {"name": "layer_0_attn", "params": 1024, "sensitivity": 0.1, "supported_bits": [4, 8, 16]},
        {"name": "layer_1_mlp", "params": 4096, "sensitivity": 0.05, "supported_bits": [4, 8, 16]},
        {"name": "layer_2_out", "params": 2048, "sensitivity": 0.9, "supported_bits": [16]}
    ]

def analyze_fp16(layer):
    if 16 in layer["supported_bits"] and layer["sensitivity"] > 0.5:
        return "high_sensitivity"
    if len(layer["supported_bits"]) == 1:
        return "unsupported_low_bit"
    return "none"

def search_budget(layers, max_bytes):
    best_config = {}
    for l in layers:
        if 8 in l["supported_bits"] and l["params"] * 1 <= max_bytes:
            best_config[l["name"]] = 8
        else:
            best_config[l["name"]] = 16
    return best_config

def calibrate_w8a8(tensor):
    abs_max = float(np.max(np.abs(tensor)))
    scale = abs_max / 127.0 if abs_max > 0 else 1.0
    zero_point = 0
    return scale, zero_point
