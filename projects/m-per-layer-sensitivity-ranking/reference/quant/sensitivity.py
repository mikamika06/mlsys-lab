import numpy as np


def compute_sensitivities(model_config, activation_stats):
    scores = []
    for layer in model_config["layers"]:
        idx = layer["index"]
        stats = activation_stats[idx]
        score = float(np.sum(stats["variance"] * (stats["mean"] ** 2)))
        scores.append(score)
    return scores
