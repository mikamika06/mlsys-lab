from memacc.accounting import layer_eager_memory, layer_sdpa_memory


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
    elif backend == "sdpa":
        fn = layer_sdpa_memory
    else:
        raise ValueError(f"Unknown backend: {backend}")

    layer_mem = fn(layer_cfg)
    return layer_mem["retained_bytes"] * n_layers


def compute_activation_size_ratio(model_cfg):
    eager_bytes = model_total_retained_bytes(model_cfg, "eager")
    sdpa_bytes = model_total_retained_bytes(model_cfg, "sdpa")
    if sdpa_bytes == 0:
        return 0.0
    return eager_bytes / sdpa_bytes
