def get_model_config():
    return {"total_vram": 16000000000, "expert_bytes": 500000000, "total_experts": 32, "base_throughput": 120.0}


def get_n_cpu_values():
    return [0, 4, 8, 16, 32]


def simulate_sweep(model_cfg, n_cpu_values):
    results = []
    for n_cpu in n_cpu_values:
        vram_bytes = max(0, model_cfg["total_vram"] - n_cpu * model_cfg["expert_bytes"])
        throughput = model_cfg["base_throughput"] * (1.0 - 0.02 * (n_cpu / model_cfg["total_experts"]))
        results.append({"n_cpu": n_cpu, "vram_bytes": vram_bytes, "throughput": throughput})
    return results


def get_placement_data():
    tensor_map = {"blk.0.attn.weight": "gpu", "blk.0.ffn.experts.0.weight": "gpu", "blk.0.ffn.experts.1.weight": "gpu"}
    overrides = {"blk.0.ffn.experts.0.weight": "cpu"}
    return tensor_map, overrides


def verify_placement(tensor_map, overrides):
    placed = {}
    for name, device in tensor_map.items():
        target = overrides.get(name, device)
        placed[name] = target
    return placed


def derive_bandwidth(target_throughput, bytes_per_token):
    return target_throughput * bytes_per_token
