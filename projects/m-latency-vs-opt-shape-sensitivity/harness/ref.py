"""Reference oracle and benchmark data generator."""

import numpy as np

GRAPH_SPEC = {
    "inputs": [
        {"name": "input_ids", "is_shape_tensor": False, "role": "execution"},
        {"name": "attention_mask", "is_shape_tensor": False, "role": "execution"},
        {"name": "kv_cache_shape", "is_shape_tensor": True, "role": "shape"},
        {"name": "output_shape", "is_shape_tensor": False, "role": "shape"},
    ]
}

WIDE_PROFILE = {
    "min": [1, 64, 512],
    "opt": [16, 64, 512],
    "max": [32, 64, 512],
}

QUERY_SHAPES = [
    [1, 64, 512],
    [2, 64, 512],
    [4, 64, 512],
    [8, 64, 512],
    [16, 64, 512],
    [24, 64, 512],
    [32, 64, 512],
]

def cost_function(shape, opt_shape):
    shape_arr = np.array(shape, dtype=np.float64)
    opt_arr = np.array(opt_shape, dtype=np.float64)
    base_cost = float(shape_arr[0] * 0.5)
    mismatch_penalty = float(np.sum((shape_arr - opt_arr) ** 2) * 0.05)
    return base_cost + mismatch_penalty

def classify_tensors_oracle(graph_spec):
    execution = []
    shape = []
    for tensor in graph_spec.get("inputs", []):
        name = tensor["name"]
        if tensor.get("is_shape_tensor") or tensor.get("role") == "shape":
            shape.append(name)
        else:
            execution.append(name)
    return {
        "execution_tensors": sorted(execution),
        "shape_tensors": sorted(shape),
    }

def compute_sensitivity_oracle(profile, cost_fn):
    min_shape = np.array(profile["min"], dtype=np.float64)
    opt_shape = np.array(profile["opt"], dtype=np.float64)
    max_shape = np.array(profile["max"], dtype=np.float64)
    c_min = cost_fn(min_shape, opt_shape)
    c_opt = cost_fn(opt_shape, opt_shape)
    c_max = cost_fn(max_shape, opt_shape)
    sens_min = float(abs(c_min - c_opt) / max(c_opt, 1e-6))
    sens_max = float(abs(c_max - c_opt) / max(c_opt, 1e-6))
    return {
        "sens_min": sens_min,
        "sens_max": sens_max,
        "total_sensitivity": float(sens_min + sens_max),
    }

def split_profile_oracle(wide_profile):
    min_s = list(wide_profile["min"])
    opt_s = list(wide_profile["opt"])
    max_s = list(wide_profile["max"])
    mid_b = (min_s[0] + max_s[0]) // 2
    opt_low_b = max(min_s[0], (min_s[0] + mid_b) // 2)
    opt_high_b = min(max_s[0], (mid_b + max_s[0]) // 2)
    return [
        {
            "min": [min_s[0]] + min_s[1:],
            "opt": [opt_low_b] + opt_s[1:],
            "max": [mid_b] + max_s[1:],
        },
        {
            "min": [mid_b] + min_s[1:],
            "opt": [opt_high_b] + opt_s[1:],
            "max": [max_s[0]] + max_s[1:],
        },
    ]

def evaluate_latency_oracle(profiles, query_shapes, cost_fn):
    total = 0.0
    for shape in query_shapes:
        s_arr = np.array(shape, dtype=np.float64)
        best = float("inf")
        for prof in profiles:
            p_min = np.array(prof["min"], dtype=np.float64)
            p_max = np.array(prof["max"], dtype=np.float64)
            if np.all(s_arr >= p_min) and np.all(s_arr <= p_max):
                c = cost_fn(s_arr, np.array(prof["opt"], dtype=np.float64))
                if c < best:
                    best = c
        if best == float("inf"):
            best = cost_fn(s_arr, np.array(profiles[0]["opt"], dtype=np.float64))
        total += best
    return float(total)
