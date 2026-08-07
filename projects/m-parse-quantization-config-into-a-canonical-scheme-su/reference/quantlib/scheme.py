import math


def parse_quantization_config(config):
    weights = config.get("config_groups", {}).get("group_0", {}).get("weights", {})
    if not weights and "weights" in config:
        weights = config["weights"]
    num_bits = weights.get("num_bits", 8)
    group_size = weights.get("group_size", -1)
    symmetric = weights.get("symmetric", True)
    strategy = weights.get("strategy", "channel")
    format_type = config.get("format", "dense")
    return {
        "num_bits": int(num_bits),
        "group_size": int(group_size),
        "symmetric": bool(symmetric),
        "strategy": str(strategy),
        "format": str(format_type)
    }


def compute_packed_shape(shape, num_bits, group_size, axis=-1):
    axis = axis % len(shape)
    elems_per_int32 = 32 // num_bits
    out_shape = list(shape)
    dim = out_shape[axis]
    if group_size > 0:
        num_groups = math.ceil(dim / group_size)
        packed_groups = math.ceil(group_size / elems_per_int32)
        out_shape[axis] = num_groups * packed_groups
    else:
        out_shape[axis] = math.ceil(dim / elems_per_int32)
    return tuple(out_shape)
