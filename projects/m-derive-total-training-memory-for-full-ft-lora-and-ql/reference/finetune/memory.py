def compute_training_memory(param_count, method, precision_bytes=2, optimizer_bytes=16):
    if method == "full":
        param_mem = param_count * precision_bytes
        grad_mem = param_count * precision_bytes
        opt_mem = param_count * optimizer_bytes
        activation_mem = param_count * 0.2
        return param_mem + grad_mem + opt_mem + activation_mem
    elif method == "lora":
        param_mem = param_count * precision_bytes
        adapter_param_count = param_count * 0.01
        grad_mem = adapter_param_count * precision_bytes
        opt_mem = adapter_param_count * optimizer_bytes
        activation_mem = param_count * 0.1
        return param_mem + grad_mem + opt_mem + activation_mem
    elif method == "qlora":
        param_mem = param_count * 0.5
        adapter_param_count = param_count * 0.01
        grad_mem = adapter_param_count * precision_bytes
        opt_mem = adapter_param_count * optimizer_bytes
        activation_mem = param_count * 0.05
        return param_mem + grad_mem + opt_mem + activation_mem
    else:
        raise ValueError(f"Unknown method: {method}")
