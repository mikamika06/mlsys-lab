import numpy as np

def calculate_size_ratio(weight, quantized_weight, scales, zeros, config):
    orig_bytes = weight.size * 2
    q_bytes = quantized_weight.size * (config.bits / 8.0)
    meta_bytes = scales.size * 4 + zeros.size * 4
    total_q_bytes = q_bytes + meta_bytes
    return float(total_q_bytes / orig_bytes)
