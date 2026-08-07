CONFIGS = [
    {"model_config": {"weight_bytes": 14 * 1024 * 1024 * 1024, "num_layers": 32, "num_kv_heads": 8, "head_dim": 128},
     "gpu_config": {"total_memory": 24 * 1024 * 1024 * 1024, "block_size": 16},
     "max_model_len": 4096},
    {"model_config": {"weight_bytes": 28 * 1024 * 1024 * 1024, "num_layers": 40, "num_kv_heads": 8, "head_dim": 128},
     "gpu_config": {"total_memory": 48 * 1024 * 1024 * 1024, "block_size": 16},
     "max_model_len": 8192},
    {"model_config": {"weight_bytes": 7 * 1024 * 1024 * 1024, "num_layers": 24, "num_kv_heads": 4, "head_dim": 128},
     "gpu_config": {"total_memory": 16 * 1024 * 1024 * 1024, "block_size": 16},
     "max_model_len": 2048},
]


def validate_capacity(model_config, gpu_config, max_model_len, gpu_memory_utilization=0.9):
    total_mem = gpu_config["total_memory"]
    available_mem = total_mem * gpu_memory_utilization
    weight_mem = model_config["weight_bytes"]
    kv_mem_available = available_mem - weight_mem
    block_size = gpu_config.get("block_size", 16)
    num_layers = model_config["num_layers"]
    num_kv_heads = model_config["num_kv_heads"]
    head_dim = model_config["head_dim"]
    bytes_per_block = num_layers * num_kv_heads * head_dim * block_size * 4
    if bytes_per_block <= 0:
        return False
    max_blocks = int(kv_mem_available // bytes_per_block)
    required_blocks_per_seq = (max_model_len + block_size - 1) // block_size
    return max_blocks >= required_blocks_per_seq * 16


def max_safe_model_len(model_config, gpu_config, gpu_memory_utilization=0.9):
    total_mem = gpu_config["total_memory"]
    available_mem = total_mem * gpu_memory_utilization
    weight_mem = model_config["weight_bytes"]
    kv_mem_available = available_mem - weight_mem
    block_size = gpu_config.get("block_size", 16)
    num_layers = model_config["num_layers"]
    num_kv_heads = model_config["num_kv_heads"]
    head_dim = model_config["head_dim"]
    bytes_per_block = num_layers * num_kv_heads * head_dim * block_size * 4
    if bytes_per_block <= 0:
        return 256
    max_blocks = int(kv_mem_available // bytes_per_block)
    concurrency = 16
    safe_len = (max_blocks * block_size) // concurrency
    return max(256, (safe_len // 256) * 256)
