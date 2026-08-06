from optmem.derive import bytes_per_param


def total_full_finetune_memory(model_params_bytes, optimizer_config):
    per_param = bytes_per_param(optimizer_config)
    num_params = model_params_bytes / 4
    return num_params * per_param
