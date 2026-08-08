def dtype_bytes(dtype_str):
    mapping = {
        "float32": 4,
        "fp32": 4,
        "float16": 2,
        "fp16": 2,
        "bfloat16": 2,
        "bf16": 2,
        "int8": 1,
    }
    return mapping[dtype_str]


def layer_eager_memory(layer_cfg):
    b = layer_cfg["batch_size"]
    s = layer_cfg["seq_len"]
    h = layer_cfg["num_heads"]
    d = layer_cfg["head_dim"]
    elem = dtype_bytes(layer_cfg.get("dtype", "float16"))

    qkv_bytes = 3 * b * s * h * d * elem
    scores_bytes = b * h * s * s * 4
    probs_bytes = b * h * s * s * elem
    ctx_bytes = b * s * h * d * elem

    retained_bytes = qkv_bytes + probs_bytes
    fwd_peak_bytes = qkv_bytes + scores_bytes + probs_bytes + ctx_bytes

    return {
        "retained_bytes": retained_bytes,
        "fwd_peak_bytes": fwd_peak_bytes,
    }


def layer_sdpa_memory(layer_cfg):
    b = layer_cfg["batch_size"]
    s = layer_cfg["seq_len"]
    h = layer_cfg["num_heads"]
    d = layer_cfg["head_dim"]
    elem = dtype_bytes(layer_cfg.get("dtype", "float16"))

    qkv_bytes = 3 * b * s * h * d * elem
    lse_bytes = b * h * s * 4
    out_bytes = b * s * h * d * elem

    retained_bytes = qkv_bytes + lse_bytes
    fwd_peak_bytes = qkv_bytes + lse_bytes + out_bytes

    return {
        "retained_bytes": retained_bytes,
        "fwd_peak_bytes": fwd_peak_bytes,
    }
