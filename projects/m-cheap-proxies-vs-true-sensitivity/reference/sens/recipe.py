import numpy as np
from sens.metric import compute_true_sensitivity


def build_recipe(layers):
    scores = [compute_true_sensitivity(l) for l in layers]
    threshold = float(np.median(scores))
    recipe = []
    for l, s in zip(layers, scores):
        bitwidth = 8 if s > threshold else 4
        recipe.append({"layer_id": l["layer_id"], "bits": bitwidth, "sensitivity": s})
    return recipe
