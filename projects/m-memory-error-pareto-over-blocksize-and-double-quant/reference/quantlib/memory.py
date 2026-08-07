def compute_footprint(layers, non_quantized, block_size, double_quant_block_size):
    total_bytes = 0
    for mod in non_quantized:
        params = mod.get("params", 0)
        bits = mod.get("bits", 16)
        total_bytes += (params * bits + 7) // 8

    for layer in layers:
        elements = layer.get("elements", 0)
        primary_bits = layer.get("primary_bits", 4)
        num_blocks = (elements + block_size - 1) // block_size
        primary_data = (elements * primary_bits + 7) // 8
        scales_bytes = num_blocks * 4
        if double_quant_block_size > 0:
            num_dq_blocks = (num_blocks + double_quant_block_size - 1) // double_quant_block_size
            dq_scales_bytes = num_dq_blocks * 1
        else:
            dq_scales_bytes = 0
        total_bytes += primary_data + scales_bytes + dq_scales_bytes

    return total_bytes
