def get_bytes_per_token(config: dict, dtype: str) -> int:
    dtype_sizes = {"float16": 2, "bfloat16": 2, "float32": 4, "int8": 1}
    b = dtype_sizes[dtype]
    layers = config["num_hidden_layers"]
    kv_heads = config.get("num_key_value_heads", config["num_attention_heads"])
    head_dim = config["hidden_size"] // config["num_attention_heads"]
    return 2 * layers * kv_heads * head_dim * b

def get_block_size_bytes(config: dict, dtype: str, block_size: int) -> int:
    return get_bytes_per_token(config, dtype) * block_size

def predict_num_gpu_blocks(config: dict, dtype: str, total_vram: int, weights_size: int, util: float, block_size: int) -> int:
    available = int(total_vram * util) - weights_size
    if available <= 0:
        return 0
    return available // get_block_size_bytes(config, dtype, block_size)

def solve_max_model_len(config: dict, dtype: str, total_vram: int, weights_size: int, util: float, block_size: int) -> int:
    return predict_num_gpu_blocks(config, dtype, total_vram, weights_size, util, block_size) * block_size

CASES = [
    {
        "config": {"num_hidden_layers": 32, "hidden_size": 4096, "num_attention_heads": 32},
        "dtype": "float16",
        "total_vram": 24 * 1024**3,
        "weights_size": 14 * 1024**3,
        "util": 0.90,
        "block_size": 16
    },
    {
        "config": {"num_hidden_layers": 32, "hidden_size": 4096, "num_attention_heads": 32, "num_key_value_heads": 8},
        "dtype": "bfloat16",
        "total_vram": 80 * 1024**3,
        "weights_size": 15 * 1024**3,
        "util": 0.95,
        "block_size": 32
    },
    {
        "config": {"num_hidden_layers": 80, "hidden_size": 8192, "num_attention_heads": 64, "num_key_value_heads": 8},
        "dtype": "float8",
        "total_vram": 40 * 1024**3,
        "weights_size": 70 * 1024**3,
        "util": 0.9,
        "block_size": 16,
        "_override_dtype": "int8"
    },
    {
        "config": {"num_hidden_layers": 40, "hidden_size": 5120, "num_attention_heads": 40, "num_key_value_heads": 10},
        "dtype": "float32",
        "total_vram": 48 * 1024**3,
        "weights_size": 20 * 1024**3,
        "util": 0.85,
        "block_size": 8
    }
]
