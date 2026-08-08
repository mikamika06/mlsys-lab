def get_bytes_per_token(config: dict, dtype: str) -> int:
    dtype_sizes = {"float16": 2, "bfloat16": 2, "float32": 4, "int8": 1}
    b = dtype_sizes[dtype]

    layers = config["num_hidden_layers"]
    kv_heads = config.get("num_key_value_heads", config["num_attention_heads"])
    head_dim = config["hidden_size"] // config["num_attention_heads"]

    return 2 * layers * kv_heads * head_dim * b


def get_block_size_bytes(config: dict, dtype: str, block_size: int) -> int:
    return get_bytes_per_token(config, dtype) * block_size
