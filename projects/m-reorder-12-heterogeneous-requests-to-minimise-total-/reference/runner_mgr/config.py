RESTART_KEYS = {"model_path", "tensor_split", "gpu_layers", "rope_scaling"}

def requires_restart(old_config, new_config):
    for k in RESTART_KEYS:
        if old_config.get(k) != new_config.get(k):
            return True
    return False
