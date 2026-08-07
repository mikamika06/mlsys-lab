import numpy as np

def build_recipe(sensitivities, budget_bits, allowed_bits):
    sorted_layers = sorted(sensitivities.keys(), key=lambda k: sensitivities[k], reverse=True)
    recipe = {}
    n_layers = len(sorted_layers)
    for i, name in enumerate(sorted_layers):
        if i < n_layers // 2:
            recipe[name] = max(allowed_bits)
        else:
            recipe[name] = min(allowed_bits)
    return recipe
