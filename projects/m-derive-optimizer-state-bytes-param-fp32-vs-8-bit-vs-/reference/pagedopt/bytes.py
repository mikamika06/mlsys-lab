def compute_optimizer_bytes(optimizer_type: str, precision: str) -> int:
    if optimizer_type == "sgd":
        if precision == "fp32":
            return 4
        elif precision == "fp16":
            return 2
        elif precision == "8-bit":
            return 1
    elif optimizer_type == "adam":
        if precision == "fp32":
            return 12
        elif precision == "fp16":
            return 8
        elif precision == "8-bit":
            return 2
    return 4
