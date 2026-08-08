"""Bit budget allocation engine."""

import numpy as np


def compute_recipe_size_bytes(recipe: dict, layer_shapes: dict) -> int:
    """Calculate model total weight size in bytes for given layer bitwidth recipe."""
    total_bits = 0
    for name, b in recipe.items():
        num_params = int(np.prod(layer_shapes[name]))
        total_bits += num_params * b
    return (total_bits + 7) // 8


def build_recipe(sensitivity_map: dict, layer_shapes: dict, budget_bytes: int, candidate_bits: list[int]) -> dict:
    """
    Construct optimal layer-to-bitwidth recipe adhering to size budget using dynamic programming.
    Returns dict: {layer_name: bitwidth}
    """
    sorted_bits = sorted(candidate_bits)
    layers = list(sensitivity_map.keys())
    layer_num_params = {name: int(np.prod(layer_shapes[name])) for name in layers}

    recipe = {name: sorted_bits[0] for name in layers}
    if compute_recipe_size_bytes(recipe, layer_shapes) > budget_bytes:
        return recipe

    min_bits = sorted_bits[0]
    scaled_budget_bits = budget_bytes * 8
    min_total_bits = sum(layer_num_params[name] * min_bits for name in layers)
    extra_bits_budget = scaled_budget_bits - min_total_bits

    if extra_bits_budget <= 0:
        return recipe

    n_layers = len(layers)
    dp = {0: (0.0, {})}

    for name in layers:
        n_params = layer_num_params[name]
        sens = sensitivity_map[name]
        min_sens = sens[min_bits]
        next_dp = {}

        for b in sorted_bits:
            added_bits = n_params * (b - min_bits)
            added_sens = sens[b] - min_sens

            for prev_b, (prev_sens_acc, prev_rec) in dp.items():
                tot_b = prev_b + added_bits
                if tot_b <= extra_bits_budget:
                    tot_s = prev_sens_acc + added_sens
                    if tot_b not in next_dp or tot_s < next_dp[tot_b][0]:
                        new_rec = dict(prev_rec)
                        new_rec[name] = b
                        next_dp[tot_b] = (tot_s, new_rec)
        if next_dp:
            dp = next_dp

    best_rec = None
    best_sens = float("inf")
    for b_used, (sens_val, rec) in dp.items():
        if sens_val < best_sens:
            best_sens = sens_val
            best_rec = rec

    if best_rec is not None:
        return best_rec
    return recipe


def find_uniform_recipe(layer_shapes: dict, budget_bytes: int, candidate_bits: list[int]) -> dict:
    """Construct uniform precision recipe that fits within size budget."""
    sorted_bits = sorted(candidate_bits, reverse=True)
    layers = list(layer_shapes.keys())
    for b in sorted_bits:
        rec = {name: b for name in layers}
        if compute_recipe_size_bytes(rec, layer_shapes) <= budget_bytes:
            return rec
    return {name: min(candidate_bits) for name in layers}
