def parse_tensors(config):
    return config["tensors"]


def is_output_tensor(name):
    return "output" in name or "lm_head" in name
