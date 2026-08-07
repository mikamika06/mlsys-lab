def oracle_weights(config):
    p = config.get("num_params", 7000000000)
    bpp = config.get("bytes_per_param", 2)
    ws = config.get("world_size", 1)
    zero = config.get("zero_stage", 0)
    return int(p * bpp / ws) if zero == 3 else int(p * bpp)

def oracle_grads(config):
    p = config.get("num_params", 7000000000)
    bpp = config.get("bytes_per_param", 2)
    ws = config.get("world_size", 1)
    zero = config.get("zero_stage", 0)
    return int(p * bpp / ws) if zero >= 2 else int(p * bpp)

def oracle_opt_states(config):
    p = config.get("num_params", 7000000000)
    ws = config.get("world_size", 1)
    zero = config.get("zero_stage", 0)
    offload = config.get("cpu_offload", False)
    if offload:
        return 0
    return int(p * 12 / ws) if zero >= 1 else int(p * 12)

def oracle_activations(config):
    h = config.get("hidden_size", 4096)
    l = config.get("num_layers", 32)
    s = config.get("seq_len", 2048)
    mb = config.get("micro_batch_size", 1)
    ckpt = config.get("activation_checkpointing", False)
    if ckpt:
        return int(34 * mb * s * h)
    return int(34 * mb * s * h * l)

def oracle_total(config):
    w = oracle_weights(config)
    g = oracle_grads(config)
    o = oracle_opt_states(config)
    a = oracle_activations(config)
    return w + g + o + a
