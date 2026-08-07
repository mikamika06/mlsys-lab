def apply_checkpointing(base_act, layers):
    raise NotImplementedError

def apply_accumulation(cfg):
    raise NotImplementedError

def apply_zero(cfg, world_size=1):
    raise NotImplementedError

def apply_offload(cfg):
    raise NotImplementedError
