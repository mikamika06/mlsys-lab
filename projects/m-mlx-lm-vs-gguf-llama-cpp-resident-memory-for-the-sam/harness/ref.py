CONFIGS = [
    {
        "model_name": "MoE-8x7B-Instruct",
        "total_params": 46700000000,
        "num_experts": 8,
        "active_experts": 2,
        "expert_size_bytes": 5000000000,
        "base_size_bytes": 7000000000,
    },
    {
        "model_name": "MoE-8x22B",
        "total_params": 141000000000,
        "num_experts": 8,
        "active_experts": 2,
        "expert_size_bytes": 15000000000,
        "base_size_bytes": 21000000000,
    },
    {
        "model_name": "MoE-16x7B",
        "total_params": 93400000000,
        "num_experts": 16,
        "active_experts": 4,
        "expert_size_bytes": 5000000000,
        "base_size_bytes": 14000000000,
    },
]


def compute_mlx_memory(cfg, quantization_bits=16):
    factor = quantization_bits / 16.0
    return int(cfg["base_size_bytes"] * factor + cfg["num_experts"] * cfg["expert_size_bytes"] * factor * 1.1)


def compute_gguf_memory(cfg, quantization_bits=4):
    factor = quantization_bits / 16.0
    return int((cfg["base_size_bytes"] + cfg["num_experts"] * cfg["expert_size_bytes"]) * factor * 1.02)


def compute_mlx_throughput(cfg):
    return float(round(120.0 / (cfg["total_params"] / 1e10), 2))


def compute_gguf_throughput(cfg):
    return float(round(145.0 / (cfg["total_params"] / 1e10), 2))


def max_resident_experts(cfg, memory_ceiling_bytes, quantization_bits=16):
    factor = quantization_bits / 16.0
    base = int(cfg["base_size_bytes"] * factor)
    expert_size = int(cfg["expert_size_bytes"] * factor)
    if memory_ceiling_bytes < base:
        return 0
    available = memory_ceiling_bytes - base
    return min(cfg["num_experts"], max(0, available // expert_size))
