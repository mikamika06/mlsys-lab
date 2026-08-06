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


def bytes_per_token(config: dict, dtype: str) -> int:
    num_layers = config["num_hidden_layers"]
    num_heads = config.get("num_key_value_heads", config["num_attention_heads"])
    if "head_dim" in config:
        head_dim = config["head_dim"]
    else:
        head_dim = config["hidden_size"] // config["num_attention_heads"]

    elem_bytes = DTYPE_SIZES[dtype.lower()]
    return 2 * num_layers * num_heads * head_dim * elem_bytes
