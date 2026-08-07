import numpy as np

MODELS = [
    {"nodes": [("aten.linear.default", "float32"), ("aten.relu.default", "float32")], "fallback_expected": ["node_0"]},
    {"nodes": [("quantized_decomposed.quantize_per_tensor.default", "int8"), ("aten.linear.default", "float32")], "fallback_expected": ["node_1"]},
]

WEIGHTS_LIST = [
    np.random.default_rng(42).normal(0.0, 1.0, (64, 64)),
    np.random.default_rng(43).normal(0.0, 0.5, (128, 128))
]

CONFIGS = [
    {"bits": 4, "group_size": 32, "has_zero_point": True},
    {"bits": 8, "group_size": 64, "has_zero_point": False}
]

def detect_fallbacks(graph_spec):
    fallbacks = []
    for i, (op, dtype) in enumerate(graph_spec["nodes"]):
        if "linear" in op and dtype == "float32":
            fallbacks.append(f"node_{i}")
    return fallbacks

def select_group_size(weights, error_budget):
    group_sizes = [32, 64, 128, 256]
    best_gs = 256
    flat = weights.flatten()
    for gs in group_sizes:
        err = 0.0
        for i in range(0, len(flat), gs):
            chunk = flat[i:i+gs]
            if len(chunk) == 0:
                continue
            quantized_approx = np.round(chunk * 7.0) / 7.0
            err += np.mean((chunk - quantized_approx) ** 2)
        if err <= error_budget and gs < best_gs:
            best_gs = gs
    return best_gs

def effective_bits(config, weight_shape):
    bits = config["bits"]
    gs = config["group_size"]
    has_zp = config["has_zero_point"]
    total_elements = np.prod(weight_shape)
    num_groups = (total_elements + gs - 1) // gs
    scale_bits = 32
    zp_bits = 8 if has_zp else 0
    overhead_bits = num_groups * (scale_bits + zp_bits)
    data_bits = total_elements * bits
    return float(data_bits + overhead_bits) / float(total_elements)
