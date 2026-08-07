def compute_adapter_size_ratio(model, adapter_state_dict):
    full_bytes = 0
    for param in model.parameters():
        full_bytes += param.numel() * param.element_size()

    adapter_bytes = 0
    for tensor in adapter_state_dict.values():
        adapter_bytes += tensor.numel() * tensor.element_size()

    if full_bytes == 0:
        return 0.0

    return adapter_bytes / full_bytes
