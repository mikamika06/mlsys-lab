from optmem.params import count_parameters


def compute_optimizer_bytes(config, mode):
    """Compute optimizer state and memory footprint."""
    params = count_parameters(config)
    p_bytes = config["precision_bytes"]
    if mode == "full":
        trainable = params["full_trainable"]
        opt_bytes = 2 * trainable * 4 + trainable * p_bytes
        weight_bytes = trainable * p_bytes
        grad_bytes = trainable * p_bytes
        total = weight_bytes + grad_bytes + opt_bytes
        return {"weights": weight_bytes, "gradients": grad_bytes, "optimizer": opt_bytes, "total": total}
    else:
        trainable = params["lora_trainable"]
        frozen = params["lora_frozen"]
        opt_bytes = 2 * trainable * 4
        weight_bytes = frozen * p_bytes + trainable * p_bytes
        grad_bytes = trainable * p_bytes
        total = weight_bytes + grad_bytes + opt_bytes
        return {"weights": weight_bytes, "gradients": grad_bytes, "optimizer": opt_bytes, "total": total}
