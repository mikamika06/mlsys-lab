from quantplan.pareto import compute_pareto


def total_footprint(weights_shape, blocksizes, double_quants, non_quantized_bytes, blocksize, double_quant):
    items = compute_pareto(weights_shape, [blocksize], [double_quant])
    return items[0]["memory_bytes"] + non_quantized_bytes
