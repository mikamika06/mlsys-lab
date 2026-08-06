def derive_optimizer_bytes_per_param(opt_type):
    if opt_type == "adam_fp32":
        return 12.0
    elif opt_type == "adam_8bit":
        return 2.0
    elif opt_type == "sgd":
        return 4.0
    else:
        raise ValueError("Unknown optimizer type")
