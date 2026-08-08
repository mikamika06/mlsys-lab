SAMPLE_MODEL = {
    "num_layers": 12,
    "batch_size": 2,
    "seq_len": 4096,
    "num_heads": 12,
    "head_dim": 64,
    "dtype": "float16",
}

TEST_LAYERS = [
    {"batch_size": 1, "seq_len": 512, "num_heads": 8, "head_dim": 64, "dtype": "float16"},
    {"batch_size": 2, "seq_len": 2048, "num_heads": 12, "head_dim": 64, "dtype": "float16"},
    {"batch_size": 4, "seq_len": 4096, "num_heads": 16, "head_dim": 128, "dtype": "bfloat16"},
]


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


def model_total_retained_bytes(model_cfg, backend):
    n_layers = model_cfg["num_layers"]
    layer_cfg = {
        "batch_size": model_cfg["batch_size"],
        "seq_len": model_cfg["seq_len"],
        "num_heads": model_cfg["num_heads"],
        "head_dim": model_cfg["head_dim"],
        "dtype": model_cfg.get("dtype", "float16"),
    }
    if backend == "eager":
        fn = layer_eager_memory
    else:
        fn = layer_sdpa_memory

    layer_mem = fn(layer_cfg)
    return layer_mem["retained_bytes"] * n_layers


def compute_activation_size_ratio(model_cfg):
    eager_bytes = model_total_retained_bytes(model_cfg, "eager")
    sdpa_bytes = model_total_retained_bytes(model_cfg, "sdpa")
    return eager_bytes / sdpa_bytes
