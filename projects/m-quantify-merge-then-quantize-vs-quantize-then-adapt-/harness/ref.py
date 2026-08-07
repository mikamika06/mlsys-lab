import numpy as np

CONFIGS = [
    {"rank": 8, "alpha": 16.0, "modules": ["q_proj", "v_proj"], "hidden_dim": 512, "dtype_bytes": 2},
    {"rank": 16, "alpha": 32.0, "modules": ["q_proj", "k_proj", "v_proj", "o_proj"], "hidden_dim": 1024, "dtype_bytes": 2},
    {"rank": 32, "alpha": 64.0, "modules": ["gate_proj", "up_proj", "down_proj"], "hidden_dim": 2048, "dtype_bytes": 2},
]

SCALING_CASES = [
    {"base_val": 1.0, "adapter_out": 0.5, "scale": 2.0, "quant_error": 0.01},
    {"base_val": 2.5, "adapter_out": -1.0, "scale": 0.5, "quant_error": 0.005},
    {"base_val": 0.0, "adapter_out": 2.0, "scale": 1.5, "quant_error": 0.02},
]

def compute_divergence(config):
    np.random.seed(42)
    dim = config["hidden_dim"]
    W = np.random.randn(dim, dim).astype(np.float32) * 0.1
    A = np.random.randn(config["rank"], dim).astype(np.float32) * 0.05
    B = np.random.randn(dim, config["rank"]).astype(np.float32) * 0.05
    delta = (B @ A) * (config["alpha"] / config["rank"])

    W_merged = W + delta
    scale_factor = np.max(np.abs(W_merged)) / 7.0
    W_merged_q = np.round(W_merged / scale_factor).clip(-8, 7) * scale_factor

    scale_base = np.max(np.abs(W)) / 7.0
    W_q = np.round(W / scale_base).clip(-8, 7) * scale_base
    W_q_adapted = W_q + delta

    diff = np.linalg.norm(W_merged_q - W_q_adapted) / (np.linalg.norm(W_q_adapted) + 1e-8)
    return float(diff)

def compute_adapter_bytes(config):
    total = 0
    dim = config["hidden_dim"]
    r = config["rank"]
    b = config["dtype_bytes"]
    for _ in config["modules"]:
        total += (dim * r * b) + (r * dim * b)
    return int(total)

def simulate_scaling_shift(case):
    base = case["base_val"] + case["quant_error"]
    adapter = case["adapter_out"] * case["scale"]
    return float(base + adapter)
