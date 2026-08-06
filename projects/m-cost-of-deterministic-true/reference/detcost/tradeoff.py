import numpy as np

def optimize_checkpointing(total_layers, memory_budget, layer_cost_fn):
    best_ckpt = []
    min_time = float("inf")
    for mask_int in range(1 << total_layers):
        ckpt = [bool((mask_int >> i) & 1) for i in range(total_layers)]
        mem = 0
        time = 0.0
        current_segment = 0
        for i in range(total_layers):
            c_time, c_mem = layer_cost_fn(i, ckpt[i])
            time += c_time
            if ckpt[i]:
                current_segment = 0
            else:
                current_segment += 1
                time += current_segment * 0.05
            mem = max(mem, c_mem + current_segment * 1024)
        if mem <= memory_budget and time < min_time:
            min_time = time
            best_ckpt = ckpt
    if not best_ckpt:
        best_ckpt = [True] * total_layers
    return best_ckpt
