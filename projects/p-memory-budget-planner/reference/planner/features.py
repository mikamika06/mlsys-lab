def apply_checkpointing(base_act, layers):
    return base_act / (layers ** 0.5)

def apply_accumulation(cfg):
    return cfg.get("accumulation_steps", 1)

def apply_zero(cfg, world_size=1):
    st = cfg.get("num_parameters", 1e9) * 8
    if cfg.get("zero_stage", 0) >= 1:
        st = st / world_size
    return {"states": st}

def apply_offload(cfg):
    st = cfg.get("num_parameters", 1e9) * 8 * 0.1
    return {"states": st}
