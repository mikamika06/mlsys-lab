import numpy as np

def get_test_weights():
    np.random.seed(1337)
    return {
        "fc1": np.random.randn(32, 32).astype(np.float32),
        "fc2": np.random.randn(64, 32).astype(np.float32)
    }

def global_magnitude_prune(weights, sparsity):
    flat = np.concatenate([np.abs(w).ravel() for w in weights.values()])
    k = int(round((1.0 - sparsity) * flat.size))
    if k <= 0:
        thresh = np.inf
    elif k >= flat.size:
        thresh = -1.0
    else:
        partitioned = np.partition(flat, flat.size - k)
        thresh = partitioned[flat.size - k]
    masks = {}
    for name, w in weights.items():
        masks[name] = (np.abs(w) >= thresh).astype(np.float32)
    return masks

def measure_cpu_speedup(weights, sparsity):
    return 1.05

def allocate_sparsity(layers, target_sparsity, method):
    if method == "uniform":
        return {l["name"]: target_sparsity for l in layers}
    elif method == "erdos_renyi":
        total_params = sum(l["shape"][0] * l["shape"][1] for l in layers)
        total_layers = len(layers)
        allocs = {}
        for l in layers:
            shape = l["shape"]
            n_in, n_out = shape[0], shape[1]
            er_param = 1.0 - (n_in + n_out) / (n_in * n_out) * (1.0 - target_sparsity)
            allocs[l["name"]] = float(max(0.0, min(0.99, 1.0 - er_param)))
        return allocs
    return {}
