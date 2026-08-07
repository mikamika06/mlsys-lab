DTYPE_BYTES = {
    "float32": 4.0,
    "float16": 2.0,
    "bfloat16": 2.0,
    "fp8": 1.0,
    "fp8_e4m3fn": 1.0,
    "fp8_e5m2": 1.0,
    "int8": 1.0,
    "int4": 0.5,
}

def get_dtype_bytes(dtype: str) -> float:
    if dtype not in DTYPE_BYTES:
        raise ValueError(f"Unknown dtype: {dtype}")
    return DTYPE_BYTES[dtype]

def per_request_kv_bytes(model_config: dict, seq_len: int, kv_dtype: str = "float16") -> int:
    num_layers = int(model_config.get("num_hidden_layers", model_config.get("num_layers", 32)))
    num_kv_heads = int(model_config.get("num_key_value_heads", model_config.get("num_attention_heads", 32)))
    hidden_size = int(model_config.get("hidden_size", 4096))
    num_attn_heads = int(model_config.get("num_attention_heads", 32))
    head_dim = int(model_config.get("head_dim", hidden_size // num_attn_heads))
    bytes_per_elem = get_dtype_bytes(kv_dtype)
    kv_bytes = 2 * num_layers * num_kv_heads * head_dim * seq_len * bytes_per_elem
    return int(kv_bytes)

def model_weights_bytes(model_config: dict, model_dtype: str = "float16") -> int:
    bytes_per_elem = get_dtype_bytes(model_dtype)
    if "num_parameters" in model_config and model_config["num_parameters"] is not None:
        return int(model_config["num_parameters"] * bytes_per_elem)
    hidden_size = int(model_config.get("hidden_size", 4096))
    num_layers = int(model_config.get("num_hidden_layers", model_config.get("num_layers", 32)))
    vocab_size = int(model_config.get("vocab_size", 32000))
    intermediate_size = int(model_config.get("intermediate_size", 4 * hidden_size))
    num_heads = int(model_config.get("num_attention_heads", 32))
    num_kv_heads = int(model_config.get("num_key_value_heads", num_heads))
    head_dim = int(model_config.get("head_dim", hidden_size // num_heads))
    embedding = vocab_size * hidden_size
    attn = hidden_size * num_heads * head_dim + 2 * hidden_size * num_kv_heads * head_dim + num_heads * head_dim * hidden_size
    mlp = 3 * hidden_size * intermediate_size
    output = hidden_size * vocab_size
    total_params = embedding + num_layers * (attn + mlp) + output
    return int(total_params * bytes_per_elem)
