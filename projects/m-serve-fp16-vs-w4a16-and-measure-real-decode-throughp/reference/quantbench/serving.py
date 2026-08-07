def calculate_model_memory(config, format_type):
    params = config["parameters"]
    if format_type == "fp16":
        return params * 2
    elif format_type == "w4a16":
        return int(params * 0.5 + params * 0.125)
    else:
        raise ValueError(f"Unknown format {format_type}")


def simulate_decode_throughput(config, format_type, batch_size):
    base_tps = config["base_throughput"] * batch_size
    if format_type == "fp16":
        return base_tps * 0.9
    elif format_type == "w4a16":
        return base_tps * 1.35
    else:
        raise ValueError(f"Unknown format {format_type}")
