import numpy as np

np.random.seed(42)

CONFIGS = []
for i in range(5):
    num_layers = 16 + i * 4
    layer_bytes = [1024 * 1024 * (10 + (j % 5)) for j in range(num_layers)]
    overhead_bytes = 50 * 1024 * 1024
    ctx_bytes = 20 * 1024 * 1024
    CONFIGS.append({
        "num_layers": num_layers,
        "layer_bytes": layer_bytes,
        "overhead_bytes": overhead_bytes,
        "ctx_bytes": ctx_bytes
    })

def compute_vram(model_spec, ngl):
    overhead = model_spec["overhead_bytes"]
    layer_sizes = model_spec["layer_bytes"]
    offloaded = sum(layer_sizes[:ngl])
    ctx = model_spec["ctx_bytes"] if ngl > 0 else 0
    return overhead + offloaded + ctx

def max_ngl_for_budget(model_spec, budget_bytes):
    best = 0
    for ngl in range(len(model_spec["layer_bytes"]) + 1):
        if compute_vram(model_spec, ngl) <= budget_bytes:
            best = ngl
    return best

def argmin_index(arr):
    return int(np.argmin(arr))

def find_throughput_knee(ngls, throughputs):
    ngls = np.array(ngls)
    tps = np.array(throughputs)
    if len(ngls) == 0:
        return 0
    x = (ngls - ngls[0]) / (ngls[-1] - ngls[0] + 1e-9)
    y = (tps - tps[0]) / (tps[-1] - tps[0] + 1e-9)
    d = np.abs(x - y) / np.sqrt(2)
    idx = argmin_index(-d)
    return int(ngls[idx])
