import numpy as np
from edge_export.joint_budget import evaluate_joint_error


def allocate_layer_strategies(layer_weights: list, target_total_bits: float, n: int, m: int, bit_options: list) -> list:
    num_layers = len(layer_weights)
    density = n / float(m)
    candidates_per_layer = []
    for w in layer_weights:
        opts = []
        for bits in bit_options:
            eff_bits = density * bits
            for use_nm in [True, False]:
                mse = evaluate_joint_error(w, n, m, bits, use_nm)
                opts.append({
                    "bits": bits,
                    "use_nm": use_nm,
                    "effective_bits": float(eff_bits),
                    "mse": mse,
                })
        candidates_per_layer.append(opts)

    best_assignment = None
    best_total_mse = float("inf")

    def search(layer_idx, current_bits, current_mse, current_assignment):
        nonlocal best_assignment, best_total_mse
        if current_bits / num_layers > target_total_bits + 1e-6:
            return
        if layer_idx == num_layers:
            if current_mse < best_total_mse:
                best_total_mse = current_mse
                best_assignment = list(current_assignment)
            return

        for opt in candidates_per_layer[layer_idx]:
            current_assignment.append(opt)
            search(layer_idx + 1, current_bits + opt["effective_bits"], current_mse + opt["mse"], current_assignment)
            current_assignment.pop()

    search(0, 0.0, 0.0, [])
    return best_assignment
