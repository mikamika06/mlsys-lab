import numpy as np

CONFIGS = [
    {
        "weights_shape": (512, 512),
        "blocksizes": [32, 64, 128],
        "double_quants": [False, True],
        "non_quantized_bytes": 10240,
        "mse_budget": 0.05
    },
    {
        "weights_shape": (1024, 256),
        "blocksizes": [64, 128],
        "double_quants": [False, True],
        "non_quantized_bytes": 20480,
        "mse_budget": 0.02
    }
]

def compute_pareto(weights_shape, blocksizes, double_quants):
    results = []
    np.random.seed(42)
    for bs in blocksizes:
        for dq in double_quants:
            base_bits = 4.0 if not dq else 3.2
            scale_overhead = (32.0 / bs) * (0.5 if dq else 1.0)
            bits = base_bits + scale_overhead
            mem = int((weights_shape[0] * weights_shape[1] * bits) / 8)
            mse = float(0.001 * (bs / 32.0) * (0.9 if dq else 1.0))
            results.append({"blocksize": bs, "double_quant": dq, "memory_bytes": mem, "mse": mse})
    return results

def best_config(weights_shape, blocksizes, double_quants, mse_budget):
    items = compute_pareto(weights_shape, blocksizes, double_quants)
    valid = [x for x in items if x["mse"] <= mse_budget]
    if not valid:
        valid = items
    mems = [x["memory_bytes"] for x in valid]
    idx = int(np.argmin(mems))
    return valid[idx]

def total_footprint(weights_shape, blocksizes, double_quants, non_quantized_bytes, blocksize, double_quant):
    items = compute_pareto(weights_shape, [blocksize], [double_quant])
    return items[0]["memory_bytes"] + non_quantized_bytes
