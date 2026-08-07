import numpy as np


def allocate_bits(weight_params, kv_elements, total_budget_bits):
    best_w_bits = 8
    best_kv_bits = 4
    min_cost_diff = float("inf")
    for w_bits in [2, 4, 8]:
        for kv_bits in [2, 4, 8]:
            cost = weight_params * w_bits + kv_elements * kv_bits
            diff = abs(cost - total_budget_bits)
            if diff < min_cost_diff:
                min_cost_diff = diff
                best_w_bits = w_bits
                best_kv_bits = kv_bits
    return {"weight_bits": best_w_bits, "kv_bits": best_kv_bits}
