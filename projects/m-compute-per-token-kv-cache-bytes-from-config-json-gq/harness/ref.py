import math

DTYPE_SIZES = {
    "float32": 4,
    "fp32": 4,
    "float16": 2,
    "fp16": 2,
    "bfloat16": 2,
    "bf16": 2,
    "int8": 1,
    "fp8": 1,
}

CONFIG_FIXTURES = [
    {
        "config": {
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 32,
            "hidden_size": 4096,
        },
        "dtype": "float16",
    },
    {
        "config": {
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "hidden_size": 4096,
        },
        "dtype": "bfloat16",
    },
    {
        "config": {
            "num_hidden_layers": 80,
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
            "hidden_size": 8192,
            "head_dim": 128,
        },
        "dtype": "fp8",
    },
    {
        "config": {
            "num_hidden_layers": 40,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "hidden_size": 5120,
            "head_dim": 128,
        },
        "dtype": "float32",
    },
]

SOLVER_FIXTURES = [
    {
        "config": CONFIG_FIXTURES[1]["config"],
        "dtype": "fp16",
        "model_weight_bytes": 14_000_000_000,
        "non_model_overhead_bytes": 2_000_000_000,
        "total_vram_bytes": 24_000_000_000,
    },
    {
        "config": CONFIG_FIXTURES[2]["config"],
        "dtype": "fp8",
        "model_weight_bytes": 70_000_000_000,
        "non_model_overhead_bytes": 4_000_000_000,
        "total_vram_bytes": 80_000_000_000,
    },
    {
        "config": CONFIG_FIXTURES[0]["config"],
        "dtype": "float16",
        "model_weight_bytes": 20_000_000_000,
        "non_model_overhead_bytes": 5_000_000_000,
        "total_vram_bytes": 16_000_000_000,
    },
]

BLOCKS_FIXTURES = [
    {
        "config": CONFIG_FIXTURES[1]["config"],
        "dtype": "bfloat16",
        "total_vram_bytes": 80_000_000_000,
        "gpu_memory_utilization": 0.90,
        "model_weight_bytes": 14_000_000_000,
        "non_model_overhead_bytes": 2_000_000_000,
        "block_size": 16,
    },
    {
        "config": CONFIG_FIXTURES[2]["config"],
        "dtype": "fp8",
        "total_vram_bytes": 40_000_000_000,
        "gpu_memory_utilization": 0.95,
        "model_weight_bytes": 35_000_000_000,
        "non_model_overhead_bytes": 1_000_000_000,
        "block_size": 32,
    },
]


def ref_bytes_per_token(config: dict, dtype: str) -> int:
    num_layers = config["num_hidden_layers"]
    num_heads = config.get("num_key_value_heads", config["num_attention_heads"])
    if "head_dim" in config:
        head_dim = config["head_dim"]
    else:
        head_dim = config["hidden_size"] // config["num_attention_heads"]
    elem_bytes = DTYPE_SIZES[dtype.lower()]
    return 2 * num_layers * num_heads * head_dim * elem_bytes


def ref_max_context_length(
    config: dict,
    dtype: str,
    model_weight_bytes: int,
    non_model_overhead_bytes: int,
    total_vram_bytes: int,
) -> int:
    bpt = ref_bytes_per_token(config, dtype)
    avail = total_vram_bytes - model_weight_bytes - non_model_overhead_bytes
    if avail <= 0 or bpt <= 0:
        return 0
    return avail // bpt


def ref_predict_num_gpu_blocks(
    config: dict,
    dtype: str,
    total_vram_bytes: int,
    gpu_memory_utilization: float,
    model_weight_bytes: int,
    non_model_overhead_bytes: int,
    block_size: int,
) -> int:
    bpt = ref_bytes_per_token(config, dtype)
    block_bytes = bpt * block_size
    usable_vram = int(math.floor(total_vram_bytes * gpu_memory_utilization))
    kv_budget = usable_vram - model_weight_bytes - non_model_overhead_bytes
    if kv_budget <= 0 or block_bytes <= 0:
        return 0
    return kv_budget // block_bytes
