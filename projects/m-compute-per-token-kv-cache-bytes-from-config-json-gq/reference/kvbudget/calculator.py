def bytes_per_element(dtype: str) -> int:
    """Return byte size for a given dtype string."""
    dt = str(dtype).lower()
    if dt in ("float32", "fp32"):
        return 4
    if dt in ("float16", "fp16", "bfloat16", "bf16"):
        return 2
    if dt in ("int8", "fp8", "fp8_e4m3fn", "fp8_e5m2"):
        return 1
    raise ValueError(f"Unsupported dtype: {dtype}")


def compute_per_token_kv_bytes(config: dict, kv_cache_dtype: str = "auto") -> int:
    """Compute KV cache bytes per token across all layers accounting for GQA."""
    num_layers = config.get("num_hidden_layers") or config.get("n_layer")
    num_heads = config.get("num_attention_heads") or config.get("n_head")
    hidden_size = config.get("hidden_size") or config.get("n_embd")
    num_kv_heads = config.get("num_key_value_heads", num_heads)

    head_dim = config.get("head_dim")
    if head_dim is None:
        head_dim = hidden_size // num_heads

    if kv_cache_dtype == "auto":
        model_dtype = config.get("torch_dtype", "float16")
        elem_bytes = bytes_per_element(model_dtype)
    else:
        elem_bytes = bytes_per_element(kv_cache_dtype)

    return 2 * num_layers * num_kv_heads * head_dim * elem_bytes
