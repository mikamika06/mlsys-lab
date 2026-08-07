def get_kernel_priority(config):
    if config["seq_len"] <= 2048 and config["head_dim"] <= 128:
        return ["flash", "math"]
    return ["math", "flash"]
