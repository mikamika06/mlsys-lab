def compute_kv_bytes(config: dict, prompt_len: int) -> int:
    num_layers = config["num_layers"]
    num_kv_heads = config["num_kv_heads"]
    head_dim = config["head_dim"]
    dtype_bytes = config.get("dtype_bytes", 2)
    return num_layers * 2 * num_kv_heads * head_dim * dtype_bytes * prompt_len

def compute_transfer_times(kv_bytes: int, bandwidths_gbps: list) -> dict:
    res = {}
    for bw in bandwidths_gbps:
        res[bw] = (kv_bytes * 8.0) / (bw * 1e9)
    return res
