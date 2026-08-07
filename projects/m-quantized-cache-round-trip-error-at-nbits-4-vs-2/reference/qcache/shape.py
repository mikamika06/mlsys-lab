def static_cache_shapes(config):
    bs = config["max_batch"]
    ml = config["max_len"]
    nh = config["num_heads"]
    hd = config["head_dim"]
    nl = config["num_layers"]
    return [(bs, nh, ml, hd) for _ in range(nl * 2)]
