from optbudget.bytes import derive_optimizer_bytes_per_param


def derive_total_memory_budget(cfg):
    trainable = cfg["trainable_params"]
    total = cfg["total_params"]
    base_bits = cfg["base_precision_bits"]
    lora_bits = cfg["lora_precision_bits"]
    base_model_bytes = total * (base_bits / 8.0)
    lora_adapter_bytes = trainable * (lora_bits / 8.0)
    gradients_bytes = cfg["gradient_bytes"]
    activations_bytes = cfg["activation_bytes"]
    opt_bytes_per_param = derive_optimizer_bytes_per_param(cfg["optimizer_type"])
    optimizer_state_bytes = trainable * opt_bytes_per_param
    total_budget = (
        base_model_bytes
        + lora_adapter_bytes
        + gradients_bytes
        + activations_bytes
        + optimizer_state_bytes
    )
    return float(total_budget)
