def allocate_sparsity(layers, target_sparsity, method):
    if method == "uniform":
        return {l["name"]: target_sparsity for l in layers}
    elif method == "erdos_renyi":
        allocs = {}
        for l in layers:
            shape = l["shape"]
            n_in, n_out = shape[0], shape[1]
            er_param = 1.0 - (n_in + n_out) / (n_in * n_out) * (1.0 - target_sparsity)
            allocs[l["name"]] = float(max(0.0, min(0.99, 1.0 - er_param)))
        return allocs
    return {}
