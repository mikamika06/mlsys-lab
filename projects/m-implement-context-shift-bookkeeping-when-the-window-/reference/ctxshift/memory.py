def kv_cache_bytes(config):
    n_layers = config["n_layers"]
    n_kv_heads = config["n_kv_heads"]
    head_dim = config["head_dim"]
    n_ctx = config["n_ctx"]
    bytes_per_element = config["bytes_per_element"]
    return n_layers * n_ctx * 2 * n_kv_heads * head_dim * bytes_per_element


def mha_vs_gqa(config):
    mha_config = dict(config)
    mha_config["n_kv_heads"] = config["n_heads"]
    mha_bytes = kv_cache_bytes(mha_config)
    gqa_bytes = kv_cache_bytes(config)
    return {"mha_bytes": mha_bytes, "gqa_bytes": gqa_bytes, "saved_bytes": mha_bytes - gqa_bytes}
