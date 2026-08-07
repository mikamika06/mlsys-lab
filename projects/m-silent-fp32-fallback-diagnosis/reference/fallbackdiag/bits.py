import numpy as np

def effective_bits(config, weight_shape):
    bits = config["bits"]
    gs = config["group_size"]
    has_zp = config["has_zero_point"]
    total_elements = np.prod(weight_shape)
    num_groups = (total_elements + gs - 1) // gs
    scale_bits = 32
    zp_bits = 8 if has_zp else 0
    overhead_bits = num_groups * (scale_bits + zp_bits)
    data_bits = total_elements * bits
    return float(data_bits + overhead_bits) / float(total_elements)
