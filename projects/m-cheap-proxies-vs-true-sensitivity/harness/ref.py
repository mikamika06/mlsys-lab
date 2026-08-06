import numpy as np


def generate_cases():
    rng = np.random.default_rng(42)
    configs = []
    for _ in range(3):
        layers = []
        for l_idx in range(4):
            w = rng.standard_normal((16, 16)).astype(np.float32)
            x = rng.standard_normal((8, 16)).astype(np.float32)
            layers.append({"layer_id": l_idx, "weight": w, "activation": x})
        configs.append(layers)
    return configs


CONFIGS = generate_cases()


def compute_proxy(layer):
    w = layer["weight"]
    return float(np.mean(np.abs(w)))


def compute_true_sensitivity(layer):
    w = layer["weight"]
    x = layer["activation"]
    out_orig = np.matmul(x, w.T)
    quant_w = np.round(w * 4.0) / 4.0
    out_quant = np.matmul(x, quant_w.T)
    return float(np.mean((out_orig - out_quant) ** 2))


def build_recipe(layers):
    scores = [compute_true_sensitivity(l) for l in layers]
    threshold = float(np.median(scores))
    recipe = []
    for l, s in zip(layers, scores):
        bitwidth = 8 if s > threshold else 4
        recipe.append({"layer_id": l["layer_id"], "bits": bitwidth, "sensitivity": s})
    return recipe
