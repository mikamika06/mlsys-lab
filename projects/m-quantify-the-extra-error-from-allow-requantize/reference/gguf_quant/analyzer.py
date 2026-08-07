import numpy as np


def quantify_requantize_error(weights, scale):
    w = np.array(weights, dtype=np.float32)
    q1 = np.round(w / scale) * scale
    q_requant = np.round(q1 / scale) * scale
    extra_error = np.mean(np.abs(w - q_requant)) - np.mean(np.abs(w - q1))
    return float(extra_error)


def compare_recipes(weights):
    w = np.array(weights, dtype=np.float32)
    default_recipe = np.mean(np.abs(w - np.round(w)))
    pure_recipe = np.mean(np.abs(w - np.round(w * 2.0) / 2.0))
    return {"default": float(default_recipe), "pure": float(pure_recipe), "diff": float(default_recipe - pure_recipe)}
