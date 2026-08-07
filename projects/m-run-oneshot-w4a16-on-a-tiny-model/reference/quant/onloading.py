import numpy as np


def simulate_execution(model_config, weights, sequential_onloading=True):
    peak_memory = 0
    base_activation_mem = model_config.get("hidden_size", 64) * 4 * 2
    if sequential_onloading:
        for name, w in weights.items():
            w_mem = w.nbytes // 4
            layer_mem = w_mem + base_activation_mem
            peak_memory = max(peak_memory, layer_mem)
    else:
        total_weight_mem = sum(w.nbytes for w in weights.values())
        peak_memory = total_weight_mem + base_activation_mem * model_config.get("num_layers", 4)
    return {"peak_memory": int(peak_memory), "sequential": bool(sequential_onloading)}
