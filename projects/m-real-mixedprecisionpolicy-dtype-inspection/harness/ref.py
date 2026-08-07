import numpy as np

POLICIES = [
    {"param_dtype": "fp32", "reduce_dtype": "fp32", "buffer_dtype": "fp32"},
    {"param_dtype": "bf16", "reduce_dtype": "fp32", "buffer_dtype": "fp32"},
    {"param_dtype": "bf16", "reduce_dtype": "bf16", "buffer_dtype": "fp32"},
]

def inspect_policy(p):
    return {
        "param_dtype": str(p.get("param_dtype")),
        "reduce_dtype": str(p.get("reduce_dtype")),
        "buffer_dtype": str(p.get("buffer_dtype")),
    }

def count_reduce_scatters(num_steps, accum_steps, use_no_sync):
    if use_no_sync:
        return num_steps // accum_steps + (1 if num_steps % accum_steps != 0 else 0)
    else:
        return num_steps

def simulate_accumulation(steps, dtype_mode):
    val = 0.0
    for _ in range(steps):
        if dtype_mode == "bf16":
            val = np.float32(val + np.float32(0.1))
        else:
            val += 0.1
    return float(val)
