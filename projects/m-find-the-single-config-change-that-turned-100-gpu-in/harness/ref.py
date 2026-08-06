import json

BASE_CFG = {"layers": 32, "offload": True, "device": "cuda", "max_gpu_layers": 32, "block_size": 128}
MOD_CFG = {"layers": 32, "offload": True, "device": "cuda", "max_gpu_layers": 16, "block_size": 128}

def find_config_change(c1, c2):
    diffs = {}
    keys = set(c1.keys()).union(set(c2.keys()))
    for k in keys:
        if c1.get(k) != c2.get(k):
            diffs[k] = (c1.get(k), c2.get(k))
    return diffs

def parse_processor_column(proc_str):
    lines = [l.strip() for l in proc_str.strip().split("\n") if l.strip()]
    gpu_count = sum(1 for l in lines if "GPU" in l.upper())
    cpu_count = sum(1 for l in lines if "CPU" in l.upper())
    total = gpu_count + cpu_count
    if total == 0:
        return {"gpu_ratio": 0.0, "gpu_layers": 0, "cpu_layers": 0, "total": 0}
    return {"gpu_ratio": gpu_count / total, "gpu_layers": gpu_count, "cpu_layers": cpu_count, "total": total}

def predict_tok_s(bandwidth_gb_s, bytes_per_token, split_ratio):
    effective_bw = bandwidth_gb_s * 1e9
    bytes_needed = bytes_per_token * (1.0 - 0.5 * split_ratio)
    return effective_bw / bytes_needed
