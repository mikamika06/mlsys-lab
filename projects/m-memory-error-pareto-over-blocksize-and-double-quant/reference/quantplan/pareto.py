def compute_pareto(weights_shape, blocksizes, double_quants):
    results = []
    for bs in blocksizes:
        for dq in double_quants:
            base_bits = 4.0 if not dq else 3.2
            scale_overhead = (32.0 / bs) * (0.5 if dq else 1.0)
            bits = base_bits + scale_overhead
            mem = int((weights_shape[0] * weights_shape[1] * bits) / 8)
            mse = float(0.001 * (bs / 32.0) * (0.9 if dq else 1.0))
            results.append({"blocksize": bs, "double_quant": dq, "memory_bytes": mem, "mse": mse})
    return results
