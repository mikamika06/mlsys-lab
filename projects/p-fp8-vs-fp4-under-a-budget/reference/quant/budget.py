def compute_effective_bpw(format_name, block_size, scale_bits):
    format_bits = {"fp8": 8, "fp6": 6, "fp4": 4, "int4": 4}.get(format_name, 8)
    scale_overhead = scale_bits / float(block_size)
    return float(format_bits) + scale_overhead
