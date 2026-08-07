RESTART_KEYS = {"model_path", "tensor_split", "gpu_layers", "rope_freq_base", "rope_scale", "vocab_size", "n_embd", "n_layer"}

def forces_restart(config_change):
    key = config_change.get("parameter")
    return key in RESTART_KEYS
