import numpy as np


def equal_size_allocation(layers, sensitivities, target_bits, allowed_bits):
    n = len(layers)
    total_elements = sum(l["size"] for l in layers)
    target_total_bits = total_elements * target_bits

    sorted_idx = np.argsort(sensitivities)[::-1]
    bits = np.full(n, min(allowed_bits), dtype=int)
    current_bits = sum(bits[i] * layers[i]["size"] for i in range(n))

    for idx in sorted_idx:
        layer_size = layers[idx]["size"]
        for b in sorted(allowed_bits, reverse=True):
            if b <= bits[idx]:
                continue
            added = (b - bits[idx]) * layer_size
            if current_bits + added <= target_total_bits:
                bits[idx] = b
                current_bits += added
                break
    return bits.tolist()


def greedy_allocation_failure_case(layers, sensitivities, budget):
    n = len(layers)
    bits = np.full(n, 16, dtype=int)
    costs = np.array([l["size"] for l in layers], dtype=int)

    sorted_idx = np.argsort(sensitivities)
    for idx in sorted_idx:
        if sum(bits * costs) > budget:
            bits[idx] = 4
        else:
            bits[idx] = 8

    if sum(bits * costs) > budget:
        excess = sum(bits * costs) - budget
        idx_max = sorted_idx[-1]
        bits[idx_max] = max(2, bits[idx_max] - 4)
    return bits.tolist()
