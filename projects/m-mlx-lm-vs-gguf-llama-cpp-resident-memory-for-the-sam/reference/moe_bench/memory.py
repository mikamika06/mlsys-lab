def estimate_resident_memory(cfg):
    mlx_mem = int(cfg["base_size_bytes"] + cfg["num_experts"] * cfg["expert_size_bytes"] * 1.1)
    gguf_mem = int((cfg["base_size_bytes"] + cfg["num_experts"] * cfg["expert_size_bytes"]) * 0.25 * 1.02)
    return mlx_mem, gguf_mem
