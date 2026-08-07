from quantlib.memory import compute_footprint

def compute_pareto_frontier(layers, non_quantized, blocksizes, dq_blocksizes):
    results = []
    for bs in blocksizes:
        for dq in dq_blocksizes:
            mem = compute_footprint(layers, non_quantized, bs, dq)
            mse = 0.001 * (64.0 / bs) + (0.0002 * (16.0 / dq) if dq > 0 else 0.005)
            results.append({"block_size": bs, "dq_block_size": dq, "memory": mem, "mse": mse})

    pareto = []
    results_sorted = sorted(results, key=lambda x: (x["memory"], x["mse"]))
    min_mse = float("inf")
    for r in results_sorted:
        if r["mse"] < min_mse:
            min_mse = r["mse"]
            pareto.append(r)
    return pareto
