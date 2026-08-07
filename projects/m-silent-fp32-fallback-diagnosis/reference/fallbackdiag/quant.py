import numpy as np

def select_group_size(weights, error_budget):
    group_sizes = [32, 64, 128, 256]
    best_gs = 256
    flat = weights.flatten()
    for gs in group_sizes:
        err = 0.0
        for i in range(0, len(flat), gs):
            chunk = flat[i:i+gs]
            if len(chunk) == 0:
                continue
            quantized_approx = np.round(chunk * 7.0) / 7.0
            err += np.mean((chunk - quantized_approx) ** 2)
        if err <= error_budget and gs < best_gs:
            best_gs = gs
    return best_gs
