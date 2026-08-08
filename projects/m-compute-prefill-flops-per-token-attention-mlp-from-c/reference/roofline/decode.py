def decode_bytes_per_step(
    config: dict, batch_size: int, context_len: int, precision_bytes: int = 2
) -> dict:
    h = config["hidden_size"]
    i = config["intermediate_size"]
    L = config["num_hidden_layers"]
    nq = config["num_attention_heads"]
    nkv = config.get("num_key_value_heads", nq)
    d = config.get("head_dim", h // nq)
    V = config["vocab_size"]

    hq = nq * d
    hkv = nkv * d

    layer_params = 2 * h * (hq + hkv) + 3 * h * i
    lm_params = h * V
    total_params = L * layer_params + lm_params

    weight_bytes = total_params * precision_bytes
    kv_elements = batch_size * context_len * 2 * L * hkv
    kv_bytes = kv_elements * precision_bytes
    total_bytes = weight_bytes + kv_bytes

    return {
        "total_bytes": float(total_bytes),
        "weight_bytes": float(weight_bytes),
        "kv_bytes": float(kv_bytes),
    }
