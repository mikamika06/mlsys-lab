import numpy as np

def repair_recipe(recipe, activations):
    fixed = dict(recipe)
    flat = np.concatenate([a.flatten() for a in activations]) if isinstance(activations, list) else activations
    mx = float(np.max(np.abs(flat)))
    scale = mx / 448.0 if mx > 0 else 1.0
    fixed["scale"] = scale
    return fixed
