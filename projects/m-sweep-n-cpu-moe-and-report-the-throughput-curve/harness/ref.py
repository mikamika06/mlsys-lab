CFG = {
    "layers": 32,
    "base_mem": 200,
    "expert_mem": 100,
    "n_experts": 8,
    "top_k": 2,
    "time_base_gpu": 2,
    "time_base_cpu": 20,
    "time_exp_gpu": 1,
    "time_exp_cpu": 15
}

def active_fraction(cfg):
    total = cfg["base_mem"] + cfg["n_experts"] * cfg["expert_mem"]
    active = cfg["base_mem"] + cfg["top_k"] * cfg["expert_mem"]
    return active / total

def vram_cost(cfg, ngl, n_cpu_experts):
    gpu_layer_cost = cfg["base_mem"] + (cfg["n_experts"] - n_cpu_experts) * cfg["expert_mem"]
    return ngl * gpu_layer_cost

def latency(cfg, ngl, n_cpu_experts):
    prob_cpu = n_cpu_experts / cfg["n_experts"]
    exp_cpu = cfg["top_k"] * prob_cpu
    exp_gpu = cfg["top_k"] - exp_cpu

    gpu_layer_time = cfg["time_base_gpu"] + exp_gpu * cfg["time_exp_gpu"] + exp_cpu * cfg["time_exp_cpu"]
    cpu_layer_time = cfg["time_base_cpu"] + cfg["top_k"] * cfg["time_exp_cpu"]

    return ngl * gpu_layer_time + (cfg["layers"] - ngl) * cpu_layer_time

def sweep_configs(cfg, max_vram):
    out = []
    for c in range(cfg["n_experts"] + 1):
        layer_cost = cfg["base_mem"] + (cfg["n_experts"] - c) * cfg["expert_mem"]
        if layer_cost <= 0:
            ngl = cfg["layers"]
        else:
            ngl = min(cfg["layers"], max_vram // layer_cost)

        lat = latency(cfg, ngl, c)
        out.append({
            "n_cpu_experts": c,
            "ngl": ngl,
            "vram": vram_cost(cfg, ngl, c),
            "latency": lat,
            "throughput": 1.0 / lat
        })
    return out
