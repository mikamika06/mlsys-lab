def calculate_kv_cache_bytes(
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    seq_len: int,
    quant_type: str = "f16",
) -> int:
    quant_type = quant_type.lower()
    num_elements = 2 * n_layers * n_kv_heads * head_dim * seq_len

    if quant_type == "f16":
        return num_elements * 2
    elif quant_type == "q8_0":
        num_blocks = (num_elements + 31) // 32
        return num_blocks * 34
    elif quant_type == "q4_0":
        num_blocks = (num_elements + 31) // 32
        return num_blocks * 18
    else:
        raise ValueError(f"Unsupported quant_type: {quant_type}")


def evaluate_perplexity_delta(
    base_ppl: float,
    quant_type: str,
    seq_len: int,
) -> float:
    quant_type = quant_type.lower()
    if quant_type == "f16":
        return 0.0

    ctx_factor = 1.0 + (seq_len / 32768.0) * 0.5
    if quant_type == "q8_0":
        return round(0.015 * ctx_factor, 4)
    elif quant_type == "q4_0":
        return round(0.180 * ctx_factor, 4)
    else:
        raise ValueError(f"Unsupported quant_type: {quant_type}")
